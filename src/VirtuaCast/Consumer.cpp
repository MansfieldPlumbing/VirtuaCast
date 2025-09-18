// File: src/VirtuaCast/Consumer.cpp

#include "Consumer.h"
#include <d3d11_1.h>
#include <d3d12.h>
#include <tlhelp32.h>
#include <chrono>

#pragma comment(lib, "d3d12.lib")

namespace VirtuaCast {

namespace {
    
    struct BroadcastManifest {
        UINT64 frameValue; UINT width; UINT height; DXGI_FORMAT format;
        LUID adapterLuid; WCHAR textureName[256]; WCHAR fenceName[256];
    };

    HANDLE GetHandleFromName_D3D12(const WCHAR* name) {
        Microsoft::WRL::ComPtr<ID3D12Device> d3d12Device;
        if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3d12Device)))) {
            return NULL;
        }
        HANDLE handle = nullptr;
        d3d12Device->OpenSharedHandleByName(name, GENERIC_ALL, &handle);
        return handle;
    }
}

struct Consumer::Impl {
    Microsoft::WRL::ComPtr<ID3D11Device> m_device;
    Microsoft::WRL::ComPtr<ID3D11Device1> m_device1;
    Microsoft::WRL::ComPtr<ID3D11Device5> m_device5;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_context;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext4> m_context4;
    LUID m_adapterLuid = {};

    bool m_isConnected = false;
    DiscoveredSharedStream m_activeStreamInfo;
    std::vector<DiscoveredSharedStream> m_discoveredStreams;

    HANDLE m_hManifest = nullptr;
    BroadcastManifest* m_pManifestView = nullptr;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_sharedTexture;
    Microsoft::WRL::ComPtr<ID3D11Fence> m_sharedFence;
    UINT64 m_lastSeenFrame = 0;

    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_privateTexture;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_privateSRV;

    UINT m_width = 0;
    UINT m_height = 0;
    
    void CreatePrivateTextureAndSRV(UINT width, UINT height, DXGI_FORMAT format);
};

Consumer::Consumer() : pImpl(std::make_unique<Impl>()) {}
Consumer::~Consumer() { Teardown(); }

HRESULT Consumer::Initialize(ID3D11Device* device, ID3D11DeviceContext* context) {
    pImpl->m_device = device;
    pImpl->m_context = context;

    HRESULT hr = pImpl->m_device.As(&pImpl->m_device1);
    if (FAILED(hr)) return hr;
    hr = pImpl->m_device.As(&pImpl->m_device5);
    if (FAILED(hr)) return hr;
    hr = pImpl->m_context.As(&pImpl->m_context4);
    if (FAILED(hr)) return hr;

    Microsoft::WRL::ComPtr<IDXGIDevice> dxgiDevice;
    pImpl->m_device.As(&dxgiDevice);
    Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
    dxgiDevice->GetAdapter(&adapter);
    DXGI_ADAPTER_DESC desc;
    adapter->GetDesc(&desc);
    pImpl->m_adapterLuid = desc.AdapterLuid;
    
    return S_OK;
}

void Consumer::Teardown() {
    Disconnect();
}

void Consumer::DiscoverStreams() {
    pImpl->m_discoveredStreams.clear();
    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hSnapshot == INVALID_HANDLE_VALUE) return;

    PROCESSENTRY32W pe32 = {};
    pe32.dwSize = sizeof(PROCESSENTRY32W);
    
    const std::vector<std::pair<std::wstring, std::wstring>> producerSignatures = {
        { L"DirectPort_Producer_Manifest_", L"DirectPort" },
    };

    if (Process32FirstW(hSnapshot, &pe32)) {
        do {
            for (const auto& sig : producerSignatures) {
                std::wstring manifestName = sig.first + std::to_wstring(pe32.th32ProcessID);
                HANDLE hManifest = OpenFileMappingW(FILE_MAP_READ, FALSE, manifestName.c_str());
                if (hManifest) {
                    BroadcastManifest* pView = (BroadcastManifest*)MapViewOfFile(hManifest, FILE_MAP_READ, 0, 0, sizeof(BroadcastManifest));
                    if (pView) {
                        if (memcmp(&pView->adapterLuid, &pImpl->m_adapterLuid, sizeof(LUID)) == 0) {
                            DiscoveredSharedStream stream;
                            stream.processId = pe32.th32ProcessID;
                            stream.processName = pe32.szExeFile;
                            stream.producerType = sig.second;
                            stream.manifestName = manifestName;
                            stream.textureName = pView->textureName;
                            stream.fenceName = pView->fenceName;
                            stream.adapterLuid = pView->adapterLuid;
                            pImpl->m_discoveredStreams.push_back(stream);
                        }
                        UnmapViewOfFile(pView);
                    }
                    CloseHandle(hManifest);
                    break;
                }
            }
        } while (Process32NextW(hSnapshot, &pe32));
    }
    CloseHandle(hSnapshot);
}


HRESULT Consumer::Connect(const DiscoveredSharedStream& stream_info) {
    Disconnect();

    pImpl->m_hManifest = OpenFileMappingW(FILE_MAP_READ, FALSE, stream_info.manifestName.c_str());
    if (!pImpl->m_hManifest) return E_FAIL;

    pImpl->m_pManifestView = (BroadcastManifest*)MapViewOfFile(pImpl->m_hManifest, FILE_MAP_READ, 0, 0, sizeof(BroadcastManifest));
    if (!pImpl->m_pManifestView) { Disconnect(); return E_FAIL; }

    HANDLE hTexture = GetHandleFromName_D3D12(pImpl->m_pManifestView->textureName);
    if (!hTexture) { Disconnect(); return E_FAIL; }
    HRESULT hr = pImpl->m_device1->OpenSharedResource1(hTexture, IID_PPV_ARGS(&pImpl->m_sharedTexture));
    CloseHandle(hTexture);
    if (FAILED(hr)) { Disconnect(); return hr; }
    
    HANDLE hFence = GetHandleFromName_D3D12(pImpl->m_pManifestView->fenceName);
    if (!hFence) { Disconnect(); return E_FAIL; }
    hr = pImpl->m_device5->OpenSharedFence(hFence, IID_PPV_ARGS(&pImpl->m_sharedFence));
    CloseHandle(hFence);
    if (FAILED(hr)) { Disconnect(); return hr; }
    
    pImpl->m_width = pImpl->m_pManifestView->width;
    pImpl->m_height = pImpl->m_pManifestView->height;
    pImpl->CreatePrivateTextureAndSRV(pImpl->m_width, pImpl->m_height, pImpl->m_pManifestView->format);
    
    pImpl->m_lastSeenFrame = 0;
    pImpl->m_isConnected = true;
    pImpl->m_activeStreamInfo = stream_info;
    
    return S_OK;
}

void Consumer::Disconnect() {
    pImpl->m_isConnected = false;
    pImpl->m_sharedFence.Reset();
    pImpl->m_sharedTexture.Reset();
    pImpl->m_privateTexture.Reset();
    pImpl->m_privateSRV.Reset();
    if (pImpl->m_pManifestView) UnmapViewOfFile(pImpl->m_pManifestView);
    if (pImpl->m_hManifest) CloseHandle(pImpl->m_hManifest);
    pImpl->m_pManifestView = nullptr;
    pImpl->m_hManifest = nullptr;
    pImpl->m_activeStreamInfo = {};
    pImpl->m_lastSeenFrame = 0;
    pImpl->m_width = 0;
    pImpl->m_height = 0;
}

ID3D11ShaderResourceView* Consumer::UpdateAndGetSRV() {
    if (!pImpl->m_isConnected) return nullptr;

    HANDLE hProcess = OpenProcess(SYNCHRONIZE, FALSE, pImpl->m_activeStreamInfo.processId);
    if (hProcess == NULL || WaitForSingleObject(hProcess, 0) != WAIT_TIMEOUT) {
        Disconnect();
        if(hProcess) CloseHandle(hProcess);
        return nullptr;
    }
    CloseHandle(hProcess);
    
    UINT64 latestFrame = pImpl->m_pManifestView->frameValue;
    if (latestFrame > pImpl->m_lastSeenFrame) {
        pImpl->m_context4->Wait(pImpl->m_sharedFence.Get(), latestFrame);
        pImpl->m_context->CopyResource(pImpl->m_privateTexture.Get(), pImpl->m_sharedTexture.Get());
        pImpl->m_lastSeenFrame = latestFrame;
    }
    return pImpl->m_privateSRV.Get();
}

void Consumer::Impl::CreatePrivateTextureAndSRV(UINT width, UINT height, DXGI_FORMAT format) {
    m_privateTexture.Reset();
    m_privateSRV.Reset();

    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    
    if (SUCCEEDED(m_device->CreateTexture2D(&desc, nullptr, &m_privateTexture))) {
        m_device->CreateShaderResourceView(m_privateTexture.Get(), nullptr, &m_privateSRV);
    }
}

bool Consumer::IsConnected() const { return pImpl->m_isConnected; }
const std::vector<DiscoveredSharedStream>& Consumer::GetDiscoveredStreams() const { return pImpl->m_discoveredStreams; }
const DiscoveredSharedStream* Consumer::GetActiveStreamInfo() const { return pImpl->m_isConnected ? &pImpl->m_activeStreamInfo : nullptr; }
std::wstring Consumer::GetProducerName() const { return pImpl->m_isConnected ? pImpl->m_activeStreamInfo.processName : L""; }
UINT Consumer::GetWidth() const { return pImpl->m_width; }
UINT Consumer::GetHeight() const { return pImpl->m_height; }

}