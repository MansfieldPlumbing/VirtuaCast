// File: src/VirtuaCast/SwapChain.h

#pragma once

#include <windows.h>
#include <d3d11_1.h>
#include <wrl/client.h>
#include <memory>

// Forward declarations
struct ID3D11Device;
struct ID3D11DeviceContext;
struct ID3D11RenderTargetView;

namespace VirtuaCast {

    class SwapChain {
    public:
        SwapChain();
        ~SwapChain();

        HRESULT Initialize(
            HWND hwnd,
            int width,
            int height,
            ID3D11Device** outDevice,
            ID3D11DeviceContext** outContext
        );

        void Teardown();
        void Resize(int newWidth, int newHeight);
        void Present();
        void ToggleFullscreen();
        bool IsFullscreen() const;

        ID3D11RenderTargetView* GetRenderTargetView() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };

} // namespace VirtuaCast