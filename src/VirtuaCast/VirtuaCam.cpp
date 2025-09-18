#ifdef NTDDI_VERSION
#undef NTDDI_VERSION
#endif
#ifdef _WIN32_WINNT
#undef _WIN32_WINNT
#endif
#define _WIN32_WINNT 0x0A00
#define NTDDI_VERSION 0x0A00000B

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11_4.h>
#include <dxgi1_2.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mferror.h>
#include <mfvirtualcamera.h>
#include <wrl/client.h>
#include <wrl/implements.h>
#include <wrl/module.h>
#include <propvarutil.h>
#include <Shlwapi.h>
#include <combaseapi.h>

#include <string>
#include <atomic>
#include <memory>
#include <mutex>

#include <initguid.h>

#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "propsys.lib")
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfsensorgroup.lib")
#pragma comment(lib, "ole32.lib")

using namespace Microsoft::WRL;

// --- Helper Functions ---
void Log(const std::wstring& msg) { WCHAR b[512]; wsprintfW(b, L"[VirtualCameraDLL] %s\n", msg.c_str()); OutputDebugStringW(b); }
void LogHRESULT(const std::wstring& msg, HRESULT hr) { WCHAR b[512]; wsprintfW(b, L"[VirtualCameraDLL] %s - HRESULT: 0x%08X\n", msg.c_str(), hr); OutputDebugStringW(b); }

// --- Manifest Definition (must match EXE) ---
struct BroadcastManifest {
    UINT64 frameValue;
    UINT width;
    UINT height;
    DXGI_FORMAT format;
    LUID adapterLuid;
    WCHAR textureName[256];
    WCHAR fenceName[256];
};

// --- GUIDs ---
DEFINE_GUID(CLSID_VirtualCameraMediaSource, 0xc629a3b3, 0xc55b, 0x4581, 0x91, 0x84, 0xb2, 0x7d, 0x5, 0x31, 0x81, 0x82);
const WCHAR* CLSID_VirtualCameraMediaSource_String = L"{C629A3B3-C55B-4581-9184-B27D05318182}";
const WCHAR* VCAM_FRIENDLY_NAME = L"VirtuaCam Media Source";

// --- Custom Interface for Initialization ---
MIDL_INTERFACE("1A2B3C4D-5E6F-7890-ABCD-EF1234567890")
IVirtualCameraControl : public IUnknown
{
public:
    virtual HRESULT STDMETHODCALLTYPE Initialize(const WCHAR* manifestName) = 0;
};

class CMFVirtualCameraMediaStream; // Forward declaration

//
// <<< THIS BLOCK IS FIXED >>>
//
class __declspec(uuid("C629A3B3-C55B-4581-9184-B27D05318182"))
CMFVirtualCameraMediaSource : public RuntimeClass<RuntimeClassFlags<ClassicCom>, IMFMediaSource, IMFMediaEventGenerator, IVirtualCameraControl>
{
public:
    CMFVirtualCameraMediaSource();
    ~CMFVirtualCameraMediaSource();

    // IMFMediaEventGenerator
    IFACEMETHOD(GetEvent)(DWORD dwFlags, IMFMediaEvent** ppEvent);
    IFACEMETHOD(BeginGetEvent)(IMFAsyncCallback* pCallback, IUnknown* punkState);
    IFACEMETHOD(EndGetEvent)(IMFAsyncResult* pResult, IMFMediaEvent** ppEvent);
    IFACEMETHOD(QueueEvent)(MediaEventType met, REFGUID guidExtendedType, HRESULT hrStatus, const PROPVARIANT* pvValue);

    // IMFMediaSource
    IFACEMETHOD(GetCharacteristics)(DWORD* pdwCharacteristics);
    IFACEMETHOD(CreatePresentationDescriptor)(IMFPresentationDescriptor** ppPresentationDescriptor);
    IFACEMETHOD(Start)(IMFPresentationDescriptor* pPresentationDescriptor, const GUID* pguidTimeFormat, const PROPVARIANT* pvarStartPosition);
    IFACEMETHOD(Stop)();
    IFACEMETHOD(Pause)();
    IFACEMETHOD(Shutdown)();

    // IVirtualCameraControl
    IFACEMETHOD(Initialize)(const WCHAR* manifestName);

private:
    HRESULT CheckShutdown() const { return m_isShutdown ? MF_E_SHUTDOWN : S_OK; }
    HRESULT CreatePresentationDescriptorInternal();

    std::mutex m_mutex;
    std::atomic<bool> m_isShutdown;

    ComPtr<IMFMediaEventQueue> m_eventQueue;
    ComPtr<IMFPresentationDescriptor> m_presentationDescriptor;
    ComPtr<CMFVirtualCameraMediaStream> m_mediaStream;
};

class CMFVirtualCameraMediaStream : public RuntimeClass<RuntimeClassFlags<ClassicCom>, IMFMediaStream, IMFMediaEventGenerator, IMFAsyncCallback>
{
public:
    CMFVirtualCameraMediaStream();
    ~CMFVirtualCameraMediaStream();

    HRESULT RuntimeClassInitialize(CMFVirtualCameraMediaSource* parent, IMFMediaEventQueue* eventQueue);
    HRESULT InitializeStream(const WCHAR* manifestName);

    // IMFMediaEventGenerator
    IFACEMETHOD(GetEvent)(DWORD dwFlags, IMFMediaEvent** ppEvent);
    IFACEMETHOD(BeginGetEvent)(IMFAsyncCallback* pCallback, IUnknown* punkState);
    IFACEMETHOD(EndGetEvent)(IMFAsyncResult* pResult, IMFMediaEvent** ppEvent);
    IFACEMETHOD(QueueEvent)(MediaEventType met, REFGUID guidExtendedType, HRESULT hrStatus, const PROPVARIANT* pvValue);

    // IMFMediaStream
    IFACEMETHOD(GetMediaSource)(IMFMediaSource** ppMediaSource);
    IFACEMETHOD(GetStreamDescriptor)(IMFStreamDescriptor** ppStreamDescriptor);
    IFACEMETHOD(RequestSample)(IUnknown* pToken);

    // IMFAsyncCallback
    IFACEMETHOD(GetParameters)(DWORD* pdwFlags, DWORD* pdwQueue);
    IFACEMETHOD(Invoke)(IMFAsyncResult* pAsyncResult);

    HRESULT StartStream();
    HRESULT StopStream();
    HRESULT Shutdown();
    ComPtr<IMFStreamDescriptor> GetStreamDescriptor() { return m_streamDescriptor; }

private:
    HRESULT CheckShutdown() const { return m_isShutdown ? MF_E_SHUTDOWN : S_OK; }
    HRESULT DeliverSample();
    HRESULT ConnectToSharedResources();
    void DisconnectSharedResources();

    std::mutex m_mutex;
    std::atomic<bool> m_isShutdown;
    std::atomic<bool> m_isStreaming;

    ComPtr<CMFVirtualCameraMediaSource> m_parent;
    ComPtr<IMFMediaEventQueue> m_eventQueue;
    ComPtr<IMFStreamDescriptor> m_streamDescriptor;
    ComPtr<IUnknown> m_sampleRequestToken;

    LONGLONG m_timeStamp;
    LONGLONG m_frameDuration;

    std::wstring m_manifestName;
    HANDLE m_hManifest;
    BroadcastManifest* m_pManifestView;
    
    ComPtr<ID3D11Device> m_pd3dDevice;
    ComPtr<ID3D11Device1> m_pd3dDevice1;
    ComPtr<ID3D11Device5> m_pd3dDevice5;
    
    ComPtr<ID3D11Texture2D> m_sharedTexture;
    ComPtr<IDXGIKeyedMutex> m_keyedMutex;
    ComPtr<ID3D11Fence> m_sharedFence;
};

// --- CMFVirtualCameraMediaSource Implementation ---
CMFVirtualCameraMediaSource::CMFVirtualCameraMediaSource() : m_isShutdown(false) {}
CMFVirtualCameraMediaSource::~CMFVirtualCameraMediaSource() { Shutdown(); }
IFACEMETHODIMP CMFVirtualCameraMediaSource::Initialize(const WCHAR* manifestName) {
    std::lock_guard<std::mutex> lock(m_mutex);
    HRESULT hr = CheckShutdown(); if (FAILED(hr)) return hr;
    if (m_mediaStream) return MF_E_ALREADY_INITIALIZED;
    hr = MFCreateEventQueue(&m_eventQueue); if (FAILED(hr)) return hr;
    hr = MakeAndInitialize<CMFVirtualCameraMediaStream>(&m_mediaStream, this, m_eventQueue.Get()); if (FAILED(hr)) return hr;
    hr = m_mediaStream->InitializeStream(manifestName); if (FAILED(hr)) return hr;
    return CreatePresentationDescriptorInternal();
}
IFACEMETHODIMP CMFVirtualCameraMediaSource::GetEvent(DWORD dwFlags, IMFMediaEvent** ppEvent) { std::lock_guard<std::mutex> lock(m_mutex); return m_eventQueue ? m_eventQueue->GetEvent(dwFlags, ppEvent) : MF_E_SHUTDOWN; }
IFACEMETHODIMP CMFVirtualCameraMediaSource::BeginGetEvent(IMFAsyncCallback* pCallback, IUnknown* punkState) { std::lock_guard<std::mutex> lock(m_mutex); return m_eventQueue ? m_eventQueue->BeginGetEvent(pCallback, punkState) : MF_E_SHUTDOWN; }
IFACEMETHODIMP CMFVirtualCameraMediaSource::EndGetEvent(IMFAsyncResult* pResult, IMFMediaEvent** ppEvent) { std::lock_guard<std::mutex> lock(m_mutex); return m_eventQueue ? m_eventQueue->EndGetEvent(pResult, ppEvent) : MF_E_SHUTDOWN; }
IFACEMETHODIMP CMFVirtualCameraMediaSource::QueueEvent(MediaEventType met, REFGUID guidExtendedType, HRESULT hrStatus, const PROPVARIANT* pvValue) { std::lock_guard<std::mutex> lock(m_mutex); return m_isShutdown ? MF_E_SHUTDOWN : m_eventQueue->QueueEventParamVar(met, guidExtendedType, hrStatus, pvValue); }
IFACEMETHODIMP CMFVirtualCameraMediaSource::GetCharacteristics(DWORD* pdwCharacteristics) { std::lock_guard<std::mutex> lock(m_mutex); if (!pdwCharacteristics) return E_POINTER; *pdwCharacteristics = MFMEDIASOURCE_IS_LIVE; return S_OK; }
IFACEMETHODIMP CMFVirtualCameraMediaSource::CreatePresentationDescriptor(IMFPresentationDescriptor** ppPresentationDescriptor) { std::lock_guard<std::mutex> lock(m_mutex); HRESULT hr = CheckShutdown(); if (FAILED(hr)) return hr; if (!m_presentationDescriptor) return MF_E_NOT_INITIALIZED; return m_presentationDescriptor->Clone(ppPresentationDescriptor); }
HRESULT CMFVirtualCameraMediaSource::CreatePresentationDescriptorInternal() { if (!m_mediaStream || !m_mediaStream->GetStreamDescriptor()) return E_UNEXPECTED; IMFStreamDescriptor* streams[] = { m_mediaStream->GetStreamDescriptor().Get() }; return MFCreatePresentationDescriptor(1, streams, &m_presentationDescriptor); }
IFACEMETHODIMP CMFVirtualCameraMediaSource::Start(IMFPresentationDescriptor* pPresentationDescriptor, const GUID* pguidTimeFormat, const PROPVARIANT* pvarStartPosition) {
    std::lock_guard<std::mutex> lock(m_mutex);
    HRESULT hr = CheckShutdown(); if (FAILED(hr)) return hr;
    if (!pPresentationDescriptor || !pvarStartPosition) return E_INVALIDARG;
    if (pguidTimeFormat && *pguidTimeFormat != GUID_NULL) return MF_E_UNSUPPORTED_TIME_FORMAT;
    DWORD streamCount = 0; hr = pPresentationDescriptor->GetStreamDescriptorCount(&streamCount); if (FAILED(hr)) return hr;
    for (DWORD i = 0; i < streamCount; ++i) { ComPtr<IMFStreamDescriptor> sd; BOOL selected = FALSE; if (SUCCEEDED(pPresentationDescriptor->GetStreamDescriptorByIndex(i, &selected, &sd)) && selected) { hr = m_mediaStream->StartStream(); break; } }
    if (SUCCEEDED(hr)) { hr = QueueEvent(MESourceStarted, GUID_NULL, hr, pvarStartPosition); }
    return hr;
}
IFACEMETHODIMP CMFVirtualCameraMediaSource::Stop() { std::lock_guard<std::mutex> lock(m_mutex); HRESULT hr = CheckShutdown(); if (FAILED(hr)) return hr; if (m_mediaStream) hr = m_mediaStream->StopStream(); if (SUCCEEDED(hr)) hr = QueueEvent(MESourceStopped, GUID_NULL, hr, nullptr); return hr; }
IFACEMETHODIMP CMFVirtualCameraMediaSource::Pause() { return MF_E_INVALID_STATE_TRANSITION; }
IFACEMETHODIMP CMFVirtualCameraMediaSource::Shutdown() {
    std::lock_guard<std::mutex> lock(m_mutex); if (m_isShutdown) return MF_E_SHUTDOWN; m_isShutdown = true;
    if (m_mediaStream) { m_mediaStream->Shutdown(); m_mediaStream.Reset(); }
    if (m_eventQueue) { m_eventQueue->Shutdown(); m_eventQueue.Reset(); }
    m_presentationDescriptor.Reset(); return S_OK;
}

// --- CMFVirtualCameraMediaStream Implementation ---
CMFVirtualCameraMediaStream::CMFVirtualCameraMediaStream() : m_isShutdown(false), m_isStreaming(false), m_parent(nullptr), m_timeStamp(0), m_frameDuration(333333), m_hManifest(nullptr), m_pManifestView(nullptr) {}
CMFVirtualCameraMediaStream::~CMFVirtualCameraMediaStream() { Shutdown(); }
HRESULT CMFVirtualCameraMediaStream::RuntimeClassInitialize(CMFVirtualCameraMediaSource* parent, IMFMediaEventQueue* eventQueue) { m_parent = parent; return eventQueue->QueryInterface(IID_PPV_ARGS(&m_eventQueue)); }
HRESULT CMFVirtualCameraMediaStream::InitializeStream(const WCHAR* manifestName) {
    std::lock_guard<std::mutex> lock(m_mutex); m_manifestName = manifestName;
    HRESULT hr = ConnectToSharedResources(); if (FAILED(hr)) { return hr; }
    ComPtr<IMFMediaType> mediaType; ComPtr<IMFMediaTypeHandler> mediaTypeHandler;
    hr = MFCreateMediaType(&mediaType);
    if(SUCCEEDED(hr)) hr = mediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    if(SUCCEEDED(hr)) hr = mediaType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_ARGB32);
    if(SUCCEEDED(hr)) hr = mediaType->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
    if(SUCCEEDED(hr)) hr = MFSetAttributeSize(mediaType.Get(), MF_MT_FRAME_SIZE, m_pManifestView->width, m_pManifestView->height);
    if(SUCCEEDED(hr)) hr = MFSetAttributeRatio(mediaType.Get(), MF_MT_FRAME_RATE, 30, 1);
    if(SUCCEEDED(hr)) hr = MFSetAttributeRatio(mediaType.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
    if (FAILED(hr)) return hr;
    IMFMediaType* mediaTypes[] = { mediaType.Get() };
    hr = MFCreateStreamDescriptor(0, 1, mediaTypes, &m_streamDescriptor); if(FAILED(hr)) return hr;
    hr = m_streamDescriptor->GetMediaTypeHandler(&mediaTypeHandler); if(SUCCEEDED(hr)) hr = mediaTypeHandler->SetCurrentMediaType(mediaType.Get());
    DisconnectSharedResources(); return hr;
}

IFACEMETHODIMP CMFVirtualCameraMediaStream::GetEvent(DWORD dwFlags, IMFMediaEvent** ppEvent) { std::lock_guard<std::mutex> lock(m_mutex); return m_eventQueue ? m_eventQueue->GetEvent(dwFlags, ppEvent) : MF_E_SHUTDOWN; }
IFACEMETHODIMP CMFVirtualCameraMediaStream::BeginGetEvent(IMFAsyncCallback* pCallback, IUnknown* punkState) { std::lock_guard<std::mutex> lock(m_mutex); return m_eventQueue ? m_eventQueue->BeginGetEvent(pCallback, punkState) : MF_E_SHUTDOWN; }
IFACEMETHODIMP CMFVirtualCameraMediaStream::EndGetEvent(IMFAsyncResult* pResult, IMFMediaEvent** ppEvent) { std::lock_guard<std::mutex> lock(m_mutex); return m_eventQueue ? m_eventQueue->EndGetEvent(pResult, ppEvent) : MF_E_SHUTDOWN; }
IFACEMETHODIMP CMFVirtualCameraMediaStream::QueueEvent(MediaEventType met, REFGUID guidExtendedType, HRESULT hrStatus, const PROPVARIANT* pvValue) { std::lock_guard<std::mutex> lock(m_mutex); return m_isShutdown ? MF_E_SHUTDOWN : m_eventQueue->QueueEventParamVar(met, guidExtendedType, hrStatus, pvValue); }
IFACEMETHODIMP CMFVirtualCameraMediaStream::GetMediaSource(IMFMediaSource** ppMediaSource) { std::lock_guard<std::mutex> lock(m_mutex); HRESULT hr = CheckShutdown(); if (FAILED(hr)) return hr; if (!ppMediaSource) return E_POINTER; *ppMediaSource = m_parent.Get(); (*ppMediaSource)->AddRef(); return S_OK; }
IFACEMETHODIMP CMFVirtualCameraMediaStream::GetStreamDescriptor(IMFStreamDescriptor** ppStreamDescriptor) { std::lock_guard<std::mutex> lock(m_mutex); HRESULT hr = CheckShutdown(); if (FAILED(hr)) return hr; if (!ppStreamDescriptor) return E_POINTER; *ppStreamDescriptor = m_streamDescriptor.Get(); (*ppStreamDescriptor)->AddRef(); return S_OK; }
IFACEMETHODIMP CMFVirtualCameraMediaStream::RequestSample(IUnknown* pToken) {
    std::lock_guard<std::mutex> lock(m_mutex); HRESULT hr = CheckShutdown(); if (FAILED(hr)) return hr;
    if (!m_isStreaming) return MF_E_MEDIA_SOURCE_WRONGSTATE;
    if (m_sampleRequestToken) return MF_E_NOTACCEPTING; 
    m_sampleRequestToken = pToken;
    hr = MFPutWorkItem(MFASYNC_CALLBACK_QUEUE_STANDARD, this, nullptr); if(FAILED(hr)) m_sampleRequestToken.Reset();
    return hr;
}
IFACEMETHODIMP CMFVirtualCameraMediaStream::GetParameters(DWORD* pdwFlags, DWORD* pdwQueue) { *pdwFlags = 0; *pdwQueue = MFASYNC_CALLBACK_QUEUE_STANDARD; return S_OK; }
IFACEMETHODIMP CMFVirtualCameraMediaStream::Invoke(IMFAsyncResult*) { return DeliverSample(); }

HRESULT CMFVirtualCameraMediaStream::DeliverSample() {
    ComPtr<IUnknown> token;
    { std::lock_guard<std::mutex> lock(m_mutex); if (m_isShutdown || !m_isStreaming) return S_OK; token = m_sampleRequestToken; m_sampleRequestToken.Reset(); }
    if (!token || !m_sharedFence || !m_keyedMutex || !m_pManifestView) return S_OK; 

    UINT64 currentFrameValue = m_pManifestView->frameValue;
    if (m_sharedFence->GetCompletedValue() < currentFrameValue) {
        HANDLE fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (fenceEvent) { m_sharedFence->SetEventOnCompletion(currentFrameValue, fenceEvent); WaitForSingleObject(fenceEvent, 200); CloseHandle(fenceEvent); }
    }
    if (FAILED(m_keyedMutex->AcquireSync(1, 16))) { RequestSample(token.Get()); return S_OK; }
    
    ComPtr<IMFSample> sample; HRESULT hr = MFCreateSample(&sample);
    if (SUCCEEDED(hr)) {
        ComPtr<IMFMediaBuffer> buffer; hr = MFCreateDXGISurfaceBuffer(__uuidof(ID3D11Texture2D), m_sharedTexture.Get(), 0, FALSE, &buffer);
        if (SUCCEEDED(hr)) { sample->AddBuffer(buffer.Get()); sample->SetSampleTime(m_timeStamp); sample->SetSampleDuration(m_frameDuration); m_timeStamp += m_frameDuration; }
    }
    m_keyedMutex->ReleaseSync(0);

    if (SUCCEEDED(hr)) {
        if (token) sample->SetUnknown(MFSampleExtension_Token, token.Get());
        PROPVARIANT pv;
        PropVariantInit(&pv);
        pv.vt = VT_UNKNOWN;
        pv.punkVal = sample.Get();
        if(pv.punkVal) pv.punkVal->AddRef();
        hr = QueueEvent(MEMediaSample, GUID_NULL, S_OK, &pv);
        PropVariantClear(&pv);
    }
    if (FAILED(hr)) { QueueEvent(MEError, GUID_NULL, hr, nullptr); }
    return S_OK;
}

HRESULT CMFVirtualCameraMediaStream::StartStream() { std::lock_guard<std::mutex> lock(m_mutex); if (m_isStreaming) return S_OK; HRESULT hr = ConnectToSharedResources(); if (FAILED(hr)) return hr; m_isStreaming = true; m_timeStamp = 0; return QueueEvent(MEStreamStarted, GUID_NULL, S_OK, nullptr); }
HRESULT CMFVirtualCameraMediaStream::StopStream() { std::lock_guard<std::mutex> lock(m_mutex); if (!m_isStreaming) return S_OK; m_isStreaming = false; m_sampleRequestToken.Reset(); DisconnectSharedResources(); return QueueEvent(MEStreamStopped, GUID_NULL, S_OK, nullptr); }
HRESULT CMFVirtualCameraMediaStream::Shutdown() {
    std::lock_guard<std::mutex> lock(m_mutex); if (m_isShutdown) return S_OK; m_isShutdown = true; m_isStreaming = false;
    m_sampleRequestToken.Reset(); if (m_eventQueue) { m_eventQueue->Shutdown(); m_eventQueue.Reset(); }
    DisconnectSharedResources(); m_streamDescriptor.Reset(); m_parent.Reset(); 
    return S_OK;
}

HRESULT CMFVirtualCameraMediaStream::ConnectToSharedResources() {
    if (m_hManifest) return S_OK;
    m_hManifest = OpenFileMappingW(FILE_MAP_READ, FALSE, m_manifestName.c_str()); if (!m_hManifest) return HRESULT_FROM_WIN32(GetLastError());
    m_pManifestView = (BroadcastManifest*)MapViewOfFile(m_hManifest, FILE_MAP_READ, 0, 0, sizeof(BroadcastManifest)); if (!m_pManifestView) { CloseHandle(m_hManifest); m_hManifest = nullptr; return HRESULT_FROM_WIN32(GetLastError()); }
    ComPtr<IDXGIFactory1> dxgiFactory; HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)); if (FAILED(hr)) return hr;
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; dxgiFactory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) { DXGI_ADAPTER_DESC1 desc; adapter->GetDesc1(&desc); if (memcmp(&desc.AdapterLuid, &m_pManifestView->adapterLuid, sizeof(LUID)) == 0) break; adapter.Reset(); }
    if (!adapter) return E_FAIL;
    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#ifdef _DEBUG
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
    hr = D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags, featureLevels, _countof(featureLevels), D3D11_SDK_VERSION, &m_pd3dDevice, nullptr, nullptr); if(FAILED(hr)) return hr;
    hr = m_pd3dDevice.As(&m_pd3dDevice1); if(FAILED(hr)) return hr;
    hr = m_pd3dDevice.As(&m_pd3dDevice5); if(FAILED(hr)) return hr;
    HANDLE sharedTextureHandle = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, m_pManifestView->textureName); if (!sharedTextureHandle) { return HRESULT_FROM_WIN32(GetLastError()); }
    hr = m_pd3dDevice1->OpenSharedResource1(sharedTextureHandle, IID_PPV_ARGS(&m_sharedTexture)); CloseHandle(sharedTextureHandle); if (FAILED(hr)) return hr;
    hr = m_sharedTexture.As(&m_keyedMutex); if (FAILED(hr)) return hr;
    HANDLE sharedFenceHandle = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, m_pManifestView->fenceName); if (!sharedFenceHandle) { return HRESULT_FROM_WIN32(GetLastError()); }
    hr = m_pd3dDevice5->OpenSharedFence(sharedFenceHandle, IID_PPV_ARGS(&m_sharedFence)); CloseHandle(sharedFenceHandle);
    return hr;
}

void CMFVirtualCameraMediaStream::DisconnectSharedResources() {
    m_keyedMutex.Reset(); m_sharedTexture.Reset(); m_sharedFence.Reset();
    m_pd3dDevice5.Reset(); m_pd3dDevice1.Reset(); m_pd3dDevice.Reset();
    if (m_pManifestView) { UnmapViewOfFile(m_pManifestView); m_pManifestView = nullptr; }
    if (m_hManifest) { CloseHandle(m_hManifest); m_hManifest = nullptr; }
}

// --- COM Registration and DLL Exports ---
HMODULE g_hModule = NULL;
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID) { if (ul_reason_for_call == DLL_PROCESS_ATTACH) { g_hModule = hModule; DisableThreadLibraryCalls(hModule); } return TRUE; }

CoCreatableClass(CMFVirtualCameraMediaSource);

STDAPI DllCanUnloadNow() { return Module<ModuleType::InProc>::GetModule().GetObjectCount() == 0 ? S_OK : S_FALSE; }
STDAPI DllGetClassObject(_In_ REFCLSID rclsid, _In_ REFIID riid, _Outptr_ LPVOID* ppv) { return Module<ModuleType::InProc>::GetModule().GetClassObject(rclsid, riid, ppv); }

STDAPI DllRegisterServer() {
    WCHAR modulePath[MAX_PATH];
    if (GetModuleFileNameW(g_hModule, modulePath, ARRAYSIZE(modulePath)) == 0) return HRESULT_FROM_WIN32(GetLastError());
    
    std::wstring keyPath = L"CLSID\\";
    keyPath += CLSID_VirtualCameraMediaSource_String;
    HKEY hKeyClsid;
    LSTATUS status = RegCreateKeyExW(HKEY_CLASSES_ROOT, keyPath.c_str(), 0, NULL, REG_OPTION_NON_VOLATILE, KEY_WRITE, NULL, &hKeyClsid, NULL);
    if (status != ERROR_SUCCESS) return HRESULT_FROM_WIN32(status);
    RegSetValueExW(hKeyClsid, NULL, 0, REG_SZ, (const BYTE*)VCAM_FRIENDLY_NAME, (wcslen(VCAM_FRIENDLY_NAME) + 1) * sizeof(WCHAR));
    
    HKEY hKeyInproc;
    status = RegCreateKeyExW(hKeyClsid, L"InprocServer32", 0, NULL, REG_OPTION_NON_VOLATILE, KEY_WRITE, NULL, &hKeyInproc, NULL);
    if (status == ERROR_SUCCESS) {
        RegSetValueExW(hKeyInproc, NULL, 0, REG_SZ, (const BYTE*)modulePath, (wcslen(modulePath) + 1) * sizeof(WCHAR));
        const WCHAR* threadingModel = L"Both";
        RegSetValueExW(hKeyInproc, L"ThreadingModel", 0, REG_SZ, (const BYTE*)threadingModel, (wcslen(threadingModel) + 1) * sizeof(WCHAR));
        RegCloseKey(hKeyInproc);
    }
    RegCloseKey(hKeyClsid);
    
    return (status == ERROR_SUCCESS) ? S_OK : HRESULT_FROM_WIN32(status);
}

STDAPI DllUnregisterServer() {
    std::wstring keyPath = L"CLSID\\";
    keyPath += CLSID_VirtualCameraMediaSource_String;
    LSTATUS status = SHDeleteKeyW(HKEY_CLASSES_ROOT, keyPath.c_str());
    return (status == ERROR_SUCCESS) ? S_OK : HRESULT_FROM_WIN32(status);
}