// File: src/VirtuaCast/CameraSource.cpp

#include "CameraSource.h"
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <sddl.h>
#include <atomic>
#include <thread>
#include <chrono>
#include <vector>
#include <d3d11_1.h>
#include <wrl/client.h>
#include <wrl/implements.h>
#include <shlwapi.h>
#include <wil/com.h>
#include <wil/result.h> 
#include <wil/resource.h>
#include <directx/d3dx12.h>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "Synchronization.lib")
#pragma comment(lib, "shlwapi.lib")

using namespace Microsoft::WRL;

// The shader now includes a simple passthrough Vertex Shader. Mirroring is handled by swapping vertex buffers.
static const char* g_shaderHLSL = R"(
cbuffer Constants : register(b0) { uint2 videoDimensions; };
struct VSInput { float3 position : POSITION; float2 uv : TEXCOORD; };
struct PSInput { float4 position : SV_POSITION; float2 uv : TEXCOORD; };
PSInput VSMain(VSInput input) { 
    PSInput output; 
    output.position = float4(input.position, 1.0f); 
    output.uv = input.uv; 
    return output; 
}
Texture2D<float4> g_textureYUY2 : register(t0);
Texture2D<float>  g_textureY    : register(t0);
Texture2D<float2> g_textureUV   : register(t1);
SamplerState      g_sampler     : register(s0);
float3 YUVtoRGB_BT709(float y, float u, float v) {
    y = (y - (16.0/255.0)) * (255.0/219.0); u = u - 0.5; v = v - 0.5;
    float r = y + 1.5748 * v; float g = y - 0.1873 * u - 0.4681 * v; float b = y + 1.8556 * u;
    return saturate(float3(r, g, b));
}
float4 PS_YUY2(PSInput input) : SV_TARGET {
    float4 yuyv = g_textureYUY2.Sample(g_sampler, input.uv); float y; float u = yuyv.y; float v = yuyv.w;
    // YUY2 stores luma for two pixels in one float4. We need to select the correct one.
    if (frac(input.uv.x * (videoDimensions.x / 2.0f)) > 0.5f) { y = yuyv.z; } else { y = yuyv.x; }
    return float4(YUVtoRGB_BT709(y, u, v), 1.0f);
}
float4 PS_NV12(PSInput input) : SV_TARGET {
    float y = g_textureY.Sample(g_sampler, input.uv).r; float2 uv = g_textureUV.Sample(g_sampler, input.uv).rg;
    return float4(YUVtoRGB_BT709(y, uv.x, uv.y), 1.0f);
}
)";

namespace VirtuaCast {

// The CaptureManager implementation remains largely the same but is now an inner class for better encapsulation.
class CaptureManager : public IMFSourceReaderCallback
{
public:
    static HRESULT CreateInstance(CaptureManager** ppManager, std::atomic<int>& write_idx, std::atomic<int>& read_idx, std::vector<BYTE> (*bufs)[3], GUID* format, long* w, long* h, std::atomic<bool>& ready)
    {
        if (!ppManager) return E_POINTER;
        *ppManager = new (std::nothrow) CaptureManager(write_idx, read_idx, bufs, format, w, h, ready);
        return (*ppManager) ? S_OK : E_OUTOFMEMORY;
    }
    STDMETHODIMP QueryInterface(REFIID iid, void** ppv) override { 
        static const QITAB qit[] = { QITABENT(CaptureManager, IMFSourceReaderCallback), { 0 } }; 
        return QISearch(this, qit, iid, ppv); 
    }
    STDMETHODIMP_(ULONG) AddRef() override { return InterlockedIncrement(&m_refCount); }
    STDMETHODIMP_(ULONG) Release() override { ULONG u = InterlockedDecrement(&m_refCount); if (!u) delete this; return u; }

    STDMETHODIMP OnReadSample(HRESULT hrStatus, DWORD, DWORD, LONGLONG, IMFSample* pSample) override;
    STDMETHODIMP OnEvent(DWORD, IMFMediaEvent*) override { return S_OK; }
    STDMETHODIMP OnFlush(DWORD) override { return S_OK; }
    void SetSourceReader(IMFSourceReader* reader) { m_sourceReader = reader; }

private:
    CaptureManager(std::atomic<int>& write, std::atomic<int>& read, std::vector<BYTE> (*bufs)[3], GUID* format, long* w, long* h, std::atomic<bool>& ready)
        : m_refCount(1), m_q_write_idx(&write), m_q_read_idx(&read), m_cpuFrameBuffers(bufs), m_videoFormat(format), m_videoWidth(w), m_videoHeight(h), m_isNewFrameAvailable(&ready) {}
    long m_refCount;
    IMFSourceReader* m_sourceReader = nullptr;
    std::atomic<int>* m_q_write_idx;
    std::atomic<int>* m_q_read_idx;
    std::vector<BYTE> (*m_cpuFrameBuffers)[3];
    GUID* m_videoFormat;
    long* m_videoWidth;
    long* m_videoHeight;
    std::atomic<bool>* m_isNewFrameAvailable;
};

STDMETHODIMP CaptureManager::OnReadSample(HRESULT hrStatus, DWORD, DWORD, LONGLONG, IMFSample* pSample)
{
    if (SUCCEEDED(hrStatus) && pSample)
    {
        ComPtr<IMFMediaBuffer> pBuffer;
        if (SUCCEEDED(pSample->ConvertToContiguousBuffer(&pBuffer)))
        {
            const UINT W = UINT(*m_videoWidth); const UINT H = UINT(*m_videoHeight);
            int writeSlot = m_q_write_idx->load(std::memory_order_relaxed);
            int nextWrite = (writeSlot + 1) % 3;
            if (nextWrite != m_q_read_idx->load(std::memory_order_acquire))
            {
                size_t expectedSize = (*m_videoFormat == MFVideoFormat_YUY2) ? (size_t(W) * H * 2) : (size_t(W) * H + (size_t(W) * H / 2));
                if ((*m_cpuFrameBuffers)[writeSlot].size() == expectedSize)
                {
                    BYTE* dst = (*m_cpuFrameBuffers)[writeSlot].data();
                    ComPtr<IMF2DBuffer2> p2DBuffer2;
                    if (SUCCEEDED(pBuffer.As(&p2DBuffer2)))
                    {
                        BYTE* pScanline0 = nullptr; LONG lPitch = 0; BYTE* pBufferStart = nullptr; DWORD cbBufferLength = 0;
                        if (SUCCEEDED(p2DBuffer2->Lock2DSize(MF2DBuffer_LockFlags_Read, &pScanline0, &lPitch, &pBufferStart, &cbBufferLength)))
                        {
                            if (*m_videoFormat == MFVideoFormat_YUY2) { for (UINT y = 0; y < H; ++y) memcpy(dst + y * W * 2, pScanline0 + y * lPitch, W * 2); }
                            else
                            {
                                BYTE* y_dst = dst;
                                BYTE* y_src = pScanline0;
                                for (UINT y = 0; y < H; ++y) {
                                    memcpy(y_dst, y_src, W);
                                    y_dst += W; y_src += lPitch;
                                }
                                BYTE* uv_dst = dst + (size_t(W) * H);
                                BYTE* uv_src = pScanline0 + (size_t(lPitch) * H);
                                for (UINT y = 0; y < H / 2; ++y) {
                                    memcpy(uv_dst, uv_src, W);
                                    uv_dst += W; uv_src += lPitch;
                                }
                            }
                            m_q_write_idx->store(nextWrite, std::memory_order_release);
                            m_isNewFrameAvailable->store(true, std::memory_order_release);
                            p2DBuffer2->Unlock2D();
                        }
                    }
                }
            }
        }
    }
    if (m_sourceReader) m_sourceReader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, nullptr, nullptr, nullptr, nullptr);
    return S_OK;
}

struct CameraSource::Impl {
    ~Impl() { if (m_isThreadRunning) Teardown(); }
    void Teardown();

    // D3D11 Interop Resources
    ComPtr<ID3D11Device5> m_d3d11Device5;
    ComPtr<ID3D11DeviceContext4> m_d3d11Context4;
    ComPtr<ID3D11Texture2D> m_sharedD3D11Texture;
    ComPtr<ID3D11ShaderResourceView> m_d3d11SRV;
    ComPtr<ID3D11Fence> m_sharedD3D11Fence;
    UINT64 m_lastWaitedFenceValue = 0;

    // Core D3D12 Resources
    ComPtr<ID3D12Device> m_d3d12Device;
    ComPtr<ID3D12CommandQueue> m_commandQueue;
    ComPtr<ID3D12CommandAllocator> m_commandAllocator;
    ComPtr<ID3D12GraphicsCommandList> m_commandList;
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12PipelineState> m_psoYUY2, m_psoNV12;
    ComPtr<ID3D12DescriptorHeap> m_srvHeap, m_rtvHeap;
    ComPtr<ID3D12Resource> m_vertexBuffer, m_vertexBufferMirrored;
    D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView, m_vertexBufferViewMirrored;
    ComPtr<ID3D12Resource> m_constantBuffer;
    uint8_t* m_pCbvDataBegin = nullptr;
    
    // Shared D3D12 Resources
    ComPtr<ID3D12Resource> m_sharedD3D12Texture;
    HANDLE m_sharedTextureHandle = nullptr;
    ComPtr<ID3D12Fence> m_sharedD3D12Fence;
    HANDLE m_sharedFenceHandle = nullptr;
    std::atomic<UINT64> m_fenceValue = 0;

    // Media Foundation Resources
    ComPtr<IMFSourceReader> m_sourceReader;
    ComPtr<CaptureManager> m_captureCallback;
    std::vector<CameraInfo> m_availableCameras;
    int m_activeCameraId = -1;
    long m_videoWidth = 0, m_videoHeight = 0;
    GUID m_videoFormat = {};
    std::vector<BYTE> m_cpuFrameBuffers[3];
    std::atomic<int> m_q_write_idx = 0, m_q_read_idx = 0;
    std::atomic<bool> m_isNewFrameAvailable = false;
    ComPtr<ID3D12Resource> m_cameraTextureY, m_cameraTextureUV, m_uploadHeap;
    bool m_mirror = false;
    
    // Threading
    std::thread m_renderThread;
    std::atomic<bool> m_isThreadRunning = false;
    
    HRESULT InitD3D12(ID3D11Device* d3d11Device);
    HRESULT LoadD3D12Assets();
    void EnumerateCameras();
    void RenderLoop();
    void PopulateCommandList();
    void WaitForGpuIdle();
};

CameraSource::CameraSource() : pImpl(std::make_unique<Impl>()) {}
CameraSource::~CameraSource() { Teardown(); }

HRESULT CameraSource::Initialize(ID3D11Device* pD3D11Device) {
    RETURN_IF_FAILED(MFStartup(MF_VERSION));
    RETURN_IF_FAILED(pImpl->InitD3D12(pD3D11Device));
    RETURN_IF_FAILED(pImpl->LoadD3D12Assets());
    pImpl->EnumerateCameras();
    return S_OK;
}

void CameraSource::Teardown() {
    DeactivateCamera();
    pImpl->Teardown();
    MFShutdown();
}

void CameraSource::Impl::WaitForGpuIdle() {
    if (!m_commandQueue || !m_d3d12Device) return;
    ComPtr<ID3D12Fence> fence;
    THROW_IF_FAILED(m_d3d12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    wil::unique_event fenceEvent(CreateEvent(nullptr, FALSE, FALSE, nullptr));
    THROW_IF_NULL_ALLOC(fenceEvent.get());

    THROW_IF_FAILED(m_commandQueue->Signal(fence.Get(), 1));
    THROW_IF_FAILED(fence->SetEventOnCompletion(1, fenceEvent.get()));
    WaitForSingleObject(fenceEvent.get(), INFINITE);
}

const std::vector<CameraInfo>& CameraSource::GetAvailableCameras() const {
    return pImpl->m_availableCameras;
}

bool CameraSource::IsActive() const {
    return pImpl->m_activeCameraId != -1;
}

void CameraSource::GetCurrentResolution(UINT& width, UINT& height) const {
    width = pImpl->m_videoWidth;
    height = pImpl->m_videoHeight;
}

void CameraSource::SetMirror(bool mirror) {
    pImpl->m_mirror = mirror;
}

HRESULT CameraSource::ActivateCamera(int cameraId, UINT width, UINT height) {
    if (cameraId < 0 || cameraId >= (int)pImpl->m_availableCameras.size()) return E_INVALIDARG;
    
    DeactivateCamera();

    const auto& camInfo = pImpl->m_availableCameras[cameraId];

    ComPtr<IMFAttributes> pAttributes;
    RETURN_IF_FAILED(MFCreateAttributes(&pAttributes, 2));
    RETURN_IF_FAILED(pAttributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID));
    RETURN_IF_FAILED(pAttributes->SetString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, camInfo.symbolic_link.c_str()));

    ComPtr<IMFMediaSource> pSource;
    RETURN_IF_FAILED(MFCreateDeviceSource(pAttributes.Get(), &pSource));

    ComPtr<IMFAttributes> pReaderAttributes;
    RETURN_IF_FAILED(MFCreateAttributes(&pReaderAttributes, 1));
    RETURN_IF_FAILED(CaptureManager::CreateInstance(&pImpl->m_captureCallback, pImpl->m_q_write_idx, pImpl->m_q_read_idx, &pImpl->m_cpuFrameBuffers, &pImpl->m_videoFormat, &pImpl->m_videoWidth, &pImpl->m_videoHeight, pImpl->m_isNewFrameAvailable));
    RETURN_IF_FAILED(pReaderAttributes->SetUnknown(MF_SOURCE_READER_ASYNC_CALLBACK, pImpl->m_captureCallback.Get()));
    
    RETURN_IF_FAILED(MFCreateSourceReaderFromMediaSource(pSource.Get(), pReaderAttributes.Get(), &pImpl->m_sourceReader));
    pImpl->m_captureCallback->SetSourceReader(pImpl->m_sourceReader.Get());

    ComPtr<IMFMediaType> pType, pBestType;
    int bestMatchScore = -10000000;

    for (DWORD i = 0; SUCCEEDED(pImpl->m_sourceReader->GetNativeMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, i, &pType)); ++i) {
        GUID subtype;
        UINT32 frameW, frameH;
        pType->GetGUID(MF_MT_SUBTYPE, &subtype);
        MFGetAttributeSize(pType.Get(), MF_MT_FRAME_SIZE, &frameW, &frameH);
        
        if (subtype == MFVideoFormat_NV12 || subtype == MFVideoFormat_YUY2) {
            int currentScore = -std::abs((long)frameW * (long)frameH - (long)width * (long)height);
            if (width == 0 && height == 0) { // If no preference, prefer 720p
                currentScore = -std::abs((long)frameW * (long)frameH - (1280*720));
            }
            if (frameW == width && frameH == height) currentScore = 100;

            if (currentScore > bestMatchScore) {
                bestMatchScore = currentScore;
                pBestType = pType;
            }
        }
        pType.Reset();
    }
    
    if (!pBestType) return E_FAIL;
    RETURN_IF_FAILED(pImpl->m_sourceReader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, NULL, pBestType.Get()));

    ComPtr<IMFMediaType> pCurrentType;
    RETURN_IF_FAILED(pImpl->m_sourceReader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, &pCurrentType));
    
    UINT32 final_width, final_height;
    MFGetAttributeSize(pCurrentType.Get(), MF_MT_FRAME_SIZE, &final_width, &final_height);
    pCurrentType->GetGUID(MF_MT_SUBTYPE, &pImpl->m_videoFormat);
    pImpl->m_videoWidth = final_width; pImpl->m_videoHeight = final_height;

    // --- Create D3D12 resources for the selected media type ---
    pImpl->m_cameraTextureY.Reset(); pImpl->m_cameraTextureUV.Reset();
    D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    D3D12_RESOURCE_DESC texDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_UNKNOWN, pImpl->m_videoWidth, pImpl->m_videoHeight, 1, 1);
    CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(pImpl->m_srvHeap->GetCPUDescriptorHandleForHeapStart());
    
    if (pImpl->m_videoFormat == MFVideoFormat_YUY2) {
        texDesc.Width = pImpl->m_videoWidth / 2;
        texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        RETURN_IF_FAILED(pImpl->m_d3d12Device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &texDesc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(&pImpl->m_cameraTextureY)));
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = texDesc.Format;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = 1;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        pImpl->m_d3d12Device->CreateShaderResourceView(pImpl->m_cameraTextureY.Get(), &srvDesc, srvHandle);
    } else { // NV12
        texDesc.Format = DXGI_FORMAT_R8_UNORM;
        RETURN_IF_FAILED(pImpl->m_d3d12Device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &texDesc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(&pImpl->m_cameraTextureY)));
        
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = texDesc.Format;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = 1;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        pImpl->m_d3d12Device->CreateShaderResourceView(pImpl->m_cameraTextureY.Get(), &srvDesc, srvHandle);

        srvHandle.Offset(1, pImpl->m_d3d12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV));
        texDesc.Width = pImpl->m_videoWidth / 2;
        texDesc.Height = pImpl->m_videoHeight / 2;
        texDesc.Format = DXGI_FORMAT_R8G8_UNORM;
        RETURN_IF_FAILED(pImpl->m_d3d12Device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &texDesc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(&pImpl->m_cameraTextureUV)));
        srvDesc.Format = texDesc.Format;
        pImpl->m_d3d12Device->CreateShaderResourceView(pImpl->m_cameraTextureUV.Get(), &srvDesc, srvHandle);
    }

    size_t frameSize = (pImpl->m_videoFormat == MFVideoFormat_YUY2) ? (size_t)final_width * final_height * 2 : (size_t)final_width * final_height * 3 / 2;
    for (int i = 0; i < 3; ++i) pImpl->m_cpuFrameBuffers[i].assign(frameSize, 0);

    texDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_B8G8R8A8_UNORM, final_width, final_height, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS);
    RETURN_IF_FAILED(pImpl->m_d3d12Device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_SHARED, &texDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&pImpl->m_sharedD3D12Texture)));
    pImpl->m_d3d12Device->CreateRenderTargetView(pImpl->m_sharedD3D12Texture.Get(), nullptr, pImpl->m_rtvHeap->GetCPUDescriptorHandleForHeapStart());
    
    wil::unique_hlocal_security_descriptor sd;
    PSECURITY_DESCRIPTOR sd_ptr = nullptr;
    ULONG sd_size = 0;
    THROW_IF_WIN32_BOOL_FALSE(ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(A;;GA;;;AU)", SDDL_REVISION_1, &sd_ptr, &sd_size));
    sd.reset(sd_ptr);
    
    SECURITY_ATTRIBUTES sa = { sizeof(sa), sd.get(), FALSE };

    RETURN_IF_FAILED(pImpl->m_d3d12Device->CreateSharedHandle(pImpl->m_sharedD3D12Texture.Get(), &sa, GENERIC_ALL, nullptr, &pImpl->m_sharedTextureHandle));
    RETURN_IF_FAILED(pImpl->m_d3d12Device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&pImpl->m_sharedD3D12Fence)));
    RETURN_IF_FAILED(pImpl->m_d3d12Device->CreateSharedHandle(pImpl->m_sharedD3D12Fence.Get(), &sa, GENERIC_ALL, nullptr, &pImpl->m_sharedFenceHandle));
    RETURN_IF_FAILED(pImpl->m_d3d11Device5->OpenSharedFence(pImpl->m_sharedFenceHandle, IID_PPV_ARGS(&pImpl->m_sharedD3D11Fence)));

    ComPtr<ID3D11Device1> d3d11Device1;
    RETURN_IF_FAILED(pImpl->m_d3d11Device5.As(&d3d11Device1));
    RETURN_IF_FAILED(d3d11Device1->OpenSharedResource1(pImpl->m_sharedTextureHandle, IID_PPV_ARGS(&pImpl->m_sharedD3D11Texture)));
    RETURN_IF_FAILED(pImpl->m_d3d11Device5->CreateShaderResourceView(pImpl->m_sharedD3D11Texture.Get(), nullptr, &pImpl->m_d3d11SRV));
    
    pImpl->m_activeCameraId = cameraId;
    pImpl->m_isThreadRunning = true;
    pImpl->m_renderThread = std::thread(&CameraSource::Impl::RenderLoop, pImpl.get());
    
    RETURN_IF_FAILED(pImpl->m_sourceReader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, nullptr, nullptr, nullptr, nullptr));
    return S_OK;
}

void CameraSource::DeactivateCamera() {
    pImpl->m_isThreadRunning = false;
    if (pImpl->m_renderThread.joinable()) {
        pImpl->m_renderThread.join();
    }
    
    pImpl->WaitForGpuIdle();
    
    pImpl->m_sourceReader.Reset();
    pImpl->m_captureCallback.Reset();
    
    pImpl->m_d3d11SRV.Reset();
    pImpl->m_sharedD3D11Texture.Reset();
    pImpl->m_sharedD3D11Fence.Reset();
    if(pImpl->m_sharedFenceHandle) { CloseHandle(pImpl->m_sharedFenceHandle); pImpl->m_sharedFenceHandle = nullptr; }
    if(pImpl->m_sharedTextureHandle) { CloseHandle(pImpl->m_sharedTextureHandle); pImpl->m_sharedTextureHandle = nullptr; }
    pImpl->m_sharedD3D12Fence.Reset();
    pImpl->m_sharedD3D12Texture.Reset();
    pImpl->m_cameraTextureY.Reset();
    pImpl->m_cameraTextureUV.Reset();
    
    pImpl->m_activeCameraId = -1;
    pImpl->m_isNewFrameAvailable = false;
    pImpl->m_q_read_idx = 0;
    pImpl->m_q_write_idx = 0;
}

ID3D11ShaderResourceView* CameraSource::GetFrameSRV() {
    if (!IsActive()) return nullptr;

    UINT64 latestFenceValue = pImpl->m_fenceValue.load(std::memory_order_acquire);
    if (latestFenceValue > pImpl->m_lastWaitedFenceValue) {
        pImpl->m_d3d11Context4->Wait(pImpl->m_sharedD3D11Fence.Get(), latestFenceValue);
        pImpl->m_lastWaitedFenceValue = latestFenceValue;
    }
    return pImpl->m_d3d11SRV.Get();
}

void CameraSource::Impl::Teardown() {
    m_isThreadRunning = false;
    if (m_renderThread.joinable()) {
        m_renderThread.join();
    }
}

void CameraSource::Impl::EnumerateCameras() {
    m_availableCameras.clear();
    ComPtr<IMFAttributes> pAttributes;
    if (FAILED(MFCreateAttributes(&pAttributes, 1))) return;
    if (FAILED(pAttributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID))) return;

    UINT32 count = 0;
    IMFActivate** devices = nullptr;
    if (SUCCEEDED(MFEnumDeviceSources(pAttributes.Get(), &devices, &count))) {
        for (UINT32 i = 0; i < count; i++) {
            wil::unique_cotaskmem_string friendlyName, symbolicLink;
            if (SUCCEEDED(devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &friendlyName, nullptr)) &&
                SUCCEEDED(devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, &symbolicLink, nullptr))) {
                m_availableCameras.push_back({(int)i, friendlyName.get(), symbolicLink.get()});
            }
            devices[i]->Release();
        }
        CoTaskMemFree(devices);
    }
}

HRESULT CameraSource::Impl::InitD3D12(ID3D11Device* d3d11Device) {
    m_d3d11Device5 = nullptr;
    m_d3d11Context4 = nullptr;
    ComPtr<ID3D11Device> tempDevice = d3d11Device;
    RETURN_IF_FAILED(tempDevice.As(&m_d3d11Device5));
    ComPtr<ID3D11DeviceContext> tempCtx;
    d3d11Device->GetImmediateContext(&tempCtx);
    RETURN_IF_FAILED(tempCtx.As(&m_d3d11Context4));

    ComPtr<IDXGIFactory4> factory;
    RETURN_IF_FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));
    RETURN_IF_FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_d3d12Device)));

    D3D12_COMMAND_QUEUE_DESC queueDesc = { D3D12_COMMAND_LIST_TYPE_DIRECT };
    RETURN_IF_FAILED(m_d3d12Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)));

    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = { D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1 };
    RETURN_IF_FAILED(m_d3d12Device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvHeap)));

    D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE };
    RETURN_IF_FAILED(m_d3d12Device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&m_srvHeap)));

    RETURN_IF_FAILED(m_d3d12Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_commandAllocator)));
    RETURN_IF_FAILED(m_d3d12Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocator.Get(), nullptr, IID_PPV_ARGS(&m_commandList)));
    m_commandList->Close();

    auto uploadHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(4096 * 2160 * 4); // Sufficiently large upload buffer
    RETURN_IF_FAILED(m_d3d12Device->CreateCommittedResource(&uploadHeapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_uploadHeap)));
    return S_OK;
}

HRESULT CameraSource::Impl::LoadD3D12Assets() {
    D3D12_DESCRIPTOR_RANGE ranges[2] = {
        { D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND },
        { D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND }
    };
    D3D12_ROOT_PARAMETER rootParameters[2] = {};
    rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    rootParameters[0].Descriptor = { 0, 0 };
    rootParameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    rootParameters[1].DescriptorTable = { _countof(ranges), ranges };

    D3D12_STATIC_SAMPLER_DESC sampler = {};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc;
    rootSignatureDesc.Init(_countof(rootParameters), rootParameters, 1, &sampler, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

    ComPtr<ID3DBlob> signature, error;
    RETURN_IF_FAILED(D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error));
    RETURN_IF_FAILED(m_d3d12Device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_rootSignature)));

    ComPtr<ID3DBlob> vsBlob, psBlobYUY2, psBlobNV12;
    RETURN_IF_FAILED(D3DCompile(g_shaderHLSL, strlen(g_shaderHLSL), nullptr, nullptr, nullptr, "VSMain", "vs_5_0", 0, 0, &vsBlob, nullptr));
    RETURN_IF_FAILED(D3DCompile(g_shaderHLSL, strlen(g_shaderHLSL), nullptr, nullptr, nullptr, "PS_YUY2", "ps_5_0", 0, 0, &psBlobYUY2, nullptr));
    RETURN_IF_FAILED(D3DCompile(g_shaderHLSL, strlen(g_shaderHLSL), nullptr, nullptr, nullptr, "PS_NV12", "ps_5_0", 0, 0, &psBlobNV12, nullptr));
    
    D3D12_INPUT_ELEMENT_DESC inputElementDescs[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.InputLayout = { inputElementDescs, _countof(inputElementDescs) };
    psoDesc.pRootSignature = m_rootSignature.Get();
    psoDesc.VS = CD3DX12_SHADER_BYTECODE(vsBlob.Get());
    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE; // Render both sides
    psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    psoDesc.DepthStencilState.DepthEnable = FALSE;
    psoDesc.DepthStencilState.StencilEnable = FALSE;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_B8G8R8A8_UNORM;
    psoDesc.SampleDesc.Count = 1;

    psoDesc.PS = CD3DX12_SHADER_BYTECODE(psBlobYUY2.Get());
    RETURN_IF_FAILED(m_d3d12Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_psoYUY2)));
    psoDesc.PS = CD3DX12_SHADER_BYTECODE(psBlobNV12.Get());
    RETURN_IF_FAILED(m_d3d12Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_psoNV12)));

    struct SimpleVertex { float Pos[3]; float Tex[2]; };
    SimpleVertex vertices[] = { {{ -1, 1, .5f }, { 0,0 }},{{ 1, 1, .5f }, { 1,0 }},{{ -1,-1,.5f }, { 0,1 }},{{ 1,-1,.5f }, { 1,1 }} };
    SimpleVertex verticesMirrored[] = { {{ -1, 1, .5f }, { 1,0 }},{{ 1, 1, .5f }, { 0,0 }},{{ -1,-1,.5f }, { 1,1 }},{{ 1,-1,.5f }, { 0,1 }} };
    
    auto uploadHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(vertices));
    RETURN_IF_FAILED(m_d3d12Device->CreateCommittedResource(&uploadHeapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_vertexBuffer)));
    RETURN_IF_FAILED(m_d3d12Device->CreateCommittedResource(&uploadHeapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_vertexBufferMirrored)));

    UINT8* pData;
    m_vertexBuffer->Map(0, nullptr, (void**)&pData); memcpy(pData, vertices, sizeof(vertices)); m_vertexBuffer->Unmap(0, nullptr);
    m_vertexBufferMirrored->Map(0, nullptr, (void**)&pData); memcpy(pData, verticesMirrored, sizeof(verticesMirrored)); m_vertexBufferMirrored->Unmap(0, nullptr);

    m_vertexBufferView = { m_vertexBuffer->GetGPUVirtualAddress(), sizeof(vertices), sizeof(SimpleVertex) };
    m_vertexBufferViewMirrored = { m_vertexBufferMirrored->GetGPUVirtualAddress(), sizeof(verticesMirrored), sizeof(SimpleVertex) };

    bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(256);
    RETURN_IF_FAILED(m_d3d12Device->CreateCommittedResource(&uploadHeapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_constantBuffer)));
    m_constantBuffer->Map(0, nullptr, reinterpret_cast<void**>(&m_pCbvDataBegin));
    
    return S_OK;
}

void CameraSource::Impl::RenderLoop() {
    while (m_isThreadRunning) {
        if (!m_isNewFrameAvailable.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(8));
            continue;
        }

        PopulateCommandList();
        ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
        m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
        
        UINT64 newFenceValue = m_fenceValue.load(std::memory_order_relaxed) + 1;
        m_commandQueue->Signal(m_sharedD3D12Fence.Get(), newFenceValue);
        m_fenceValue.store(newFenceValue, std::memory_order_release);
        m_isNewFrameAvailable = false;
    }
}

void CameraSource::Impl::PopulateCommandList() {
    m_commandAllocator->Reset();
    m_commandList->Reset(m_commandAllocator.Get(), nullptr);

    const int read_slot = m_q_read_idx.load(std::memory_order_relaxed);
    if (read_slot != m_q_write_idx.load(std::memory_order_acquire)) {
        const BYTE* frame = m_cpuFrameBuffers[read_slot].data();
        const UINT W = m_videoWidth; const UINT H = m_videoHeight;
        bool isNV12 = m_videoFormat == MFVideoFormat_NV12;

        if (isNV12) {
            auto preBarriers = { 
                CD3DX12_RESOURCE_BARRIER::Transition(m_cameraTextureY.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
                CD3DX12_RESOURCE_BARRIER::Transition(m_cameraTextureUV.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST)
            };
            m_commandList->ResourceBarrier((UINT)preBarriers.size(), preBarriers.begin());
        } else {
            m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_cameraTextureY.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST));
        }

        if (isNV12) {
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT yLayout = {}, uvLayout = {};
            D3D12_RESOURCE_DESC yDesc = m_cameraTextureY->GetDesc(), uvDesc = m_cameraTextureUV->GetDesc();
            UINT64 ySize = 0;
            m_d3d12Device->GetCopyableFootprints(&yDesc, 0, 1, 0, &yLayout, nullptr, nullptr, &ySize);
            m_d3d12Device->GetCopyableFootprints(&uvDesc, 0, 1, ySize, &uvLayout, nullptr, nullptr, nullptr);
            
            uint8_t* uploadPtr;
            m_uploadHeap->Map(0, nullptr, (void**)&uploadPtr);
            for (UINT y = 0; y < H; ++y) memcpy(uploadPtr + yLayout.Offset + y * yLayout.Footprint.RowPitch, frame + y * W, W);
            for (UINT y = 0; y < H / 2; ++y) memcpy(uploadPtr + uvLayout.Offset + y * uvLayout.Footprint.RowPitch, frame + (size_t)W * H + y * W, W);
            m_uploadHeap->Unmap(0, nullptr);

            m_commandList->CopyTextureRegion(&CD3DX12_TEXTURE_COPY_LOCATION(m_cameraTextureY.Get(), 0), 0, 0, 0, &CD3DX12_TEXTURE_COPY_LOCATION(m_uploadHeap.Get(), yLayout), nullptr);
            m_commandList->CopyTextureRegion(&CD3DX12_TEXTURE_COPY_LOCATION(m_cameraTextureUV.Get(), 0), 0, 0, 0, &CD3DX12_TEXTURE_COPY_LOCATION(m_uploadHeap.Get(), uvLayout), nullptr);
        } else {
            D3D12_SUBRESOURCE_DATA yuy2Data = { frame, W * 2, (UINT_PTR)W * H * 2 };
            UpdateSubresources(m_commandList.Get(), m_cameraTextureY.Get(), m_uploadHeap.Get(), 0, 0, 1, &yuy2Data);
        }
        
        if (isNV12) {
            auto postBarriers = {
                 CD3DX12_RESOURCE_BARRIER::Transition(m_cameraTextureY.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
                 CD3DX12_RESOURCE_BARRIER::Transition(m_cameraTextureUV.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)
            };
            m_commandList->ResourceBarrier((UINT)postBarriers.size(), postBarriers.begin());
        } else {
             m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_cameraTextureY.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
        }
        
        m_q_read_idx.store((read_slot + 1) % 3, std::memory_order_release);
    }
    
    m_commandList->SetGraphicsRootSignature(m_rootSignature.Get());
    m_commandList->SetGraphicsRootConstantBufferView(0, m_constantBuffer->GetGPUVirtualAddress());
    ID3D12DescriptorHeap* heaps[] = { m_srvHeap.Get() };
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);

    struct CbData { UINT dims[2]; float pad[2]; };
    CbData cb = { {(UINT)m_videoWidth, (UINT)m_videoHeight} };
    memcpy(m_pCbvDataBegin, &cb, sizeof(cb));

    m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_sharedD3D12Texture.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET));
    
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    m_commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
    const float clearColor[] = { 0.0f, 0.0f, 0.1f, 1.0f };
    m_commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
    
    D3D12_VIEWPORT vp = { 0.f, 0.f, (float)m_videoWidth, (float)m_videoHeight, 0.f, 1.f };
    D3D12_RECT sr = { 0, 0, m_videoWidth, m_videoHeight };
    m_commandList->RSSetViewports(1, &vp);
    m_commandList->RSSetScissorRects(1, &sr);

    m_commandList->SetPipelineState((m_videoFormat == MFVideoFormat_YUY2) ? m_psoYUY2.Get() : m_psoNV12.Get());
    m_commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    m_commandList->IASetVertexBuffers(0, 1, m_mirror ? &m_vertexBufferViewMirrored : &m_vertexBufferView);
    m_commandList->SetGraphicsRootDescriptorTable(1, m_srvHeap->GetGPUDescriptorHandleForHeapStart());
    m_commandList->DrawInstanced(4, 1, 0, 0);

    m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_sharedD3D12Texture.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON));
    
    m_commandList->Close();
}

}