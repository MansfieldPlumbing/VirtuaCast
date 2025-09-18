// File: src/VirtuaCast/DMLHelper.h
#pragma once

#include <memory>
#include <windows.h>

namespace Ort {
    class SessionOptions;
}

struct ID3D11Device;
struct ID3D12Device;
struct ID3D12CommandQueue;
struct IDMLDevice;

namespace VirtuaCast {

    class DMLHelper {
    public:
        DMLHelper();
        ~DMLHelper();

        HRESULT Initialize(ID3D11Device* d3d11Device);
        void Teardown();
        void ConfigureSessionOptions(Ort::SessionOptions& options);
        void WaitForGpuIdle();

        ID3D12Device* GetD3D12Device() const;
        ID3D12CommandQueue* GetCommandQueue() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };

}