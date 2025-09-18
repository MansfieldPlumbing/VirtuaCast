// File: src/VirtuaCast/SwapChain.cpp

#include "SwapChain.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

using namespace Microsoft::WRL;

namespace VirtuaCast {

struct SwapChain::Impl {
    ComPtr<ID3D11Device> m_device;
    ComPtr<ID3D11DeviceContext> m_context;
    ComPtr<IDXGISwapChain1> m_swapChain;
    ComPtr<ID3D11RenderTargetView> m_rtv;
    HWND m_hwnd = nullptr;
    bool m_isFullscreen = false;

    void CreateRenderTarget();
    void CleanupRenderTarget();
};

SwapChain::SwapChain() : pImpl(std::make_unique<Impl>()) {}
SwapChain::~SwapChain() { Teardown(); }

void SwapChain::Impl::CleanupRenderTarget() {
    if (m_rtv) {
        m_rtv.Reset();
    }
}

void SwapChain::Impl::CreateRenderTarget() {
    ComPtr<ID3D11Texture2D> pBackBuffer;
    if (m_swapChain) {
        m_swapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
        if (pBackBuffer) {
            m_device->CreateRenderTargetView(pBackBuffer.Get(), NULL, &m_rtv);
        }
    }
}

HRESULT SwapChain::Initialize(HWND hwnd, int width, int height, ID3D11Device** outDevice, ID3D11DeviceContext** outContext) {
    pImpl->m_hwnd = hwnd;
    DXGI_SWAP_CHAIN_DESC1 sd = {};
    sd.Width = width;
    sd.Height = height;
    sd.BufferCount = 2;
    sd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sd.SampleDesc.Count = 1;
    
    UINT createDeviceFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#ifdef _DEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_FEATURE_LEVEL featureLevel;
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags,
        nullptr, 0, D3D11_SDK_VERSION, &pImpl->m_device, &featureLevel, &pImpl->m_context);
    if (FAILED(hr)) return hr;

    ComPtr<IDXGIDevice> dxgiDevice;
    hr = pImpl->m_device.As(&dxgiDevice);
    if (FAILED(hr)) return hr;

    ComPtr<IDXGIAdapter> dxgiAdapter;
    hr = dxgiDevice->GetAdapter(&dxgiAdapter);
    if (FAILED(hr)) return hr;

    ComPtr<IDXGIFactory2> dxgiFactory;
    hr = dxgiAdapter->GetParent(IID_PPV_ARGS(&dxgiFactory));
    if (FAILED(hr)) return hr;

    hr = dxgiFactory->CreateSwapChainForHwnd(pImpl->m_device.Get(), pImpl->m_hwnd, &sd, nullptr, nullptr, &pImpl->m_swapChain);
    if (FAILED(hr)) return hr;
    
    dxgiFactory->MakeWindowAssociation(pImpl->m_hwnd, DXGI_MWA_NO_WINDOW_CHANGES | DXGI_MWA_NO_ALT_ENTER);

    pImpl->CreateRenderTarget();

    *outDevice = pImpl->m_device.Get();
    if (*outDevice) (*outDevice)->AddRef();
    *outContext = pImpl->m_context.Get();
    if (*outContext) (*outContext)->AddRef();

    return S_OK;
}

void SwapChain::Teardown() {
    pImpl->CleanupRenderTarget();
    if (pImpl->m_swapChain) {
        pImpl->m_swapChain->SetFullscreenState(FALSE, NULL);
    }
    pImpl->m_swapChain.Reset();
}

void SwapChain::Resize(int newWidth, int newHeight) {
    if (!pImpl->m_swapChain) return;
    pImpl->CleanupRenderTarget();
    pImpl->m_swapChain->ResizeBuffers(0, newWidth, newHeight, DXGI_FORMAT_UNKNOWN, 0);
    pImpl->CreateRenderTarget();
}

void SwapChain::ToggleFullscreen() {
    if (pImpl->m_swapChain) {
        pImpl->m_isFullscreen = !pImpl->m_isFullscreen;
        pImpl->m_swapChain->SetFullscreenState(pImpl->m_isFullscreen, nullptr);
    }
}

bool SwapChain::IsFullscreen() const {
    return pImpl->m_isFullscreen;
}

void SwapChain::Present() {
    if (pImpl->m_swapChain) {
        pImpl->m_swapChain->Present(1, 0);
    }
}

ID3D11RenderTargetView* SwapChain::GetRenderTargetView() const {
    return pImpl->m_rtv.Get();
}

} // namespace VirtuaCast