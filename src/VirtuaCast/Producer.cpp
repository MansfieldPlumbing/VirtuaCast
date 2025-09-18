// File: src/VirtuaCast/Producer.cpp

#include "Producer.h"
#include <dxgi1_2.h>
#include <sddl.h>
#include <vector>

#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "advapi32.lib")

using namespace Microsoft::WRL;

namespace {
    struct BroadcastManifest {
        UINT64 frameValue;
        UINT width;
        UINT height;
        DXGI_FORMAT format;
        LUID adapterLuid;
        WCHAR textureName[256];
        WCHAR fenceName[256];
    };
}

namespace VirtuaCast {

struct Producer::Impl {
    ComPtr<ID3D11Device5> m_device5;
    ComPtr<ID3D11DeviceContext4> m_context4;
    ComPtr<ID3D11Texture2D> m_sharedTexture;
    ComPtr<ID3D11Fence> m_sharedFence;

    HANDLE m_sharedTextureHandle = nullptr;
    HANDLE m_sharedFenceHandle = nullptr;
    UINT64 m_frameValue = 0;
    
    HANDLE m_hManifest = nullptr;
    BroadcastManifest* m_pManifestView = nullptr;
    std::wstring m_sharedTextureName, m_sharedFenceName;

    HRESULT CreateSharedFence();
    HRESULT CreateBroadcastManifest();
};

Producer::Producer() : pImpl(std::make_unique<Impl>()) {}
Producer::~Producer() { Teardown(); }

HRESULT Producer::Initialize(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11Texture2D* sharedTexture) {
    if (!device || !context || !sharedTexture) return E_INVALIDARG;

    HRESULT hr = device->QueryInterface(IID_PPV_ARGS(&pImpl->m_device5));
    if (FAILED(hr)) return hr;

    hr = context->QueryInterface(IID_PPV_ARGS(&pImpl->m_context4));
    if (FAILED(hr)) return hr;

    pImpl->m_sharedTexture = sharedTexture;

    DWORD pid = GetCurrentProcessId();
    pImpl->m_sharedTextureName = L"Global\\VirtuaCast_PreviewTexture_" + std::to_wstring(pid);
    pImpl->m_sharedFenceName = L"Global\\VirtuaCast_PreviewFence_" + std::to_wstring(pid);

    ComPtr<IDXGIResource1> dxgiResource;
    hr = pImpl->m_sharedTexture.As(&dxgiResource);
    if (FAILED(hr)) return hr;

    PSECURITY_DESCRIPTOR sd = nullptr;
    if (ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(A;;GA;;;AU)", SDDL_REVISION_1, &sd, NULL)) {
        SECURITY_ATTRIBUTES sa{ sizeof(sa), sd, FALSE };
        hr = dxgiResource->CreateSharedHandle(&sa, GENERIC_ALL, pImpl->m_sharedTextureName.c_str(), &pImpl->m_sharedTextureHandle);
        LocalFree(sd);
        if (FAILED(hr)) return hr;
    }
    else {
        return E_FAIL;
    }
    
    hr = pImpl->CreateSharedFence();
    if (FAILED(hr)) return hr;

    hr = pImpl->CreateBroadcastManifest();
    if (FAILED(hr)) return hr;

    return S_OK;
}

void Producer::Teardown() {
    if (pImpl->m_pManifestView) UnmapViewOfFile(pImpl->m_pManifestView);
    if (pImpl->m_hManifest) CloseHandle(pImpl->m_hManifest);
    if (pImpl->m_sharedTextureHandle) CloseHandle(pImpl->m_sharedTextureHandle);
    if (pImpl->m_sharedFenceHandle) CloseHandle(pImpl->m_sharedFenceHandle);

    pImpl->m_pManifestView = nullptr;
    pImpl->m_hManifest = nullptr;
    pImpl->m_sharedTextureHandle = nullptr;
    pImpl->m_sharedFenceHandle = nullptr;

    pImpl->m_sharedFence.Reset();
    pImpl->m_sharedTexture.Reset();
    pImpl->m_context4.Reset();
    pImpl->m_device5.Reset();
}

void Producer::PublishFrame() {
    if (!pImpl->m_sharedFence || !pImpl->m_context4) return;

    pImpl->m_frameValue++;
    pImpl->m_context4->Signal(pImpl->m_sharedFence.Get(), pImpl->m_frameValue);

    if (pImpl->m_pManifestView) {
        InterlockedExchange64(reinterpret_cast<volatile LONGLONG*>(&pImpl->m_pManifestView->frameValue), pImpl->m_frameValue);
    }
}

HRESULT Producer::Impl::CreateSharedFence() {
    HRESULT hr = m_device5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&m_sharedFence));
    if (FAILED(hr)) return hr;

    PSECURITY_DESCRIPTOR sd = nullptr;
    if (ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(A;;GA;;;AU)", SDDL_REVISION_1, &sd, NULL)) {
        SECURITY_ATTRIBUTES sa{ sizeof(sa), sd, FALSE };
        hr = m_sharedFence->CreateSharedHandle(&sa, GENERIC_ALL, m_sharedFenceName.c_str(), &m_sharedFenceHandle);
        LocalFree(sd);
        if (FAILED(hr)) return hr;
    } else {
        return E_FAIL;
    }
    return S_OK;
}

HRESULT Producer::Impl::CreateBroadcastManifest() {
    LUID adapterLuid = {};
    ComPtr<IDXGIDevice> dxgiDevice;
    if (SUCCEEDED(m_device5.As(&dxgiDevice))) {
        ComPtr<IDXGIAdapter> adapter;
        if (SUCCEEDED(dxgiDevice->GetAdapter(&adapter))) {
            DXGI_ADAPTER_DESC desc;
            if (SUCCEEDED(adapter->GetDesc(&desc))) { adapterLuid = desc.AdapterLuid; }
        }
    }

    std::wstring manifestName = L"VirtuaCast_Preview_Manifest_" + std::to_wstring(GetCurrentProcessId());
    
    PSECURITY_DESCRIPTOR sd = nullptr;
    if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(A;;GA;;;AU)", SDDL_REVISION_1, &sd, NULL)) return E_FAIL;
    
    SECURITY_ATTRIBUTES sa = { sizeof(sa), sd, FALSE };
    m_hManifest = CreateFileMappingW(INVALID_HANDLE_VALUE, &sa, PAGE_READWRITE, 0, sizeof(BroadcastManifest), manifestName.c_str());
    LocalFree(sd);

    if (m_hManifest == NULL) return E_FAIL;
    
    m_pManifestView = (BroadcastManifest*)MapViewOfFile(m_hManifest, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(BroadcastManifest));
    if (m_pManifestView == nullptr) {
        CloseHandle(m_hManifest);
        m_hManifest = nullptr;
        return E_FAIL;
    }
    
    D3D11_TEXTURE2D_DESC desc;
    m_sharedTexture->GetDesc(&desc);

    ZeroMemory(m_pManifestView, sizeof(BroadcastManifest));
    m_pManifestView->width = desc.Width;
    m_pManifestView->height = desc.Height;
    m_pManifestView->format = desc.Format;
    m_pManifestView->adapterLuid = adapterLuid;
    wcscpy_s(m_pManifestView->textureName, _countof(m_pManifestView->textureName), m_sharedTextureName.c_str());
    wcscpy_s(m_pManifestView->fenceName, _countof(m_pManifestView->fenceName), m_sharedFenceName.c_str());
    
    return S_OK;
}

}