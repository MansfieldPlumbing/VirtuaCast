// File: src/VirtuaCast/DMLHelper.cpp
#include "DMLHelper.h"
#include <d3d11.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <stdexcept>
#include <DirectML.h>
#include "onnxruntime_cxx_api.h"
#include <dml_provider_factory.h>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "directml.lib")

using namespace Microsoft::WRL;

namespace VirtuaCast {

struct DMLHelper::Impl {
    ComPtr<ID3D12Device> d3d12Device;
    ComPtr<IDMLDevice> dmlDevice;
    ComPtr<ID3D12CommandQueue> commandQueue;
};

DMLHelper::DMLHelper() : pImpl(std::make_unique<Impl>()) {}
DMLHelper::~DMLHelper() { Teardown(); }

HRESULT DMLHelper::Initialize(ID3D11Device* d3d11Device) {
    if (!d3d11Device) return E_INVALIDARG;

    ComPtr<IDXGIDevice> dxgiDevice;
    HRESULT hr = d3d11Device->QueryInterface(IID_PPV_ARGS(&dxgiDevice));
    if (FAILED(hr)) return hr;

    ComPtr<IDXGIAdapter> dxgiAdapter;
    hr = dxgiDevice->GetAdapter(&dxgiAdapter);
    if (FAILED(hr)) return hr;

    hr = D3D12CreateDevice(dxgiAdapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&pImpl->d3d12Device));
    if (FAILED(hr)) return hr;

    hr = DMLCreateDevice(pImpl->d3d12Device.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&pImpl->dmlDevice));
    if (FAILED(hr)) return hr;

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    hr = pImpl->d3d12Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&pImpl->commandQueue));
    
    return hr;
}

void DMLHelper::Teardown() { pImpl.reset(); }

void DMLHelper::ConfigureSessionOptions(Ort::SessionOptions& options) {
    if (pImpl && pImpl->dmlDevice && pImpl->commandQueue) {
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProviderEx_DML(options, pImpl->dmlDevice.Get(), pImpl->commandQueue.Get()));
    } else {
        throw std::runtime_error("DirectML module was not initialized or DML/D3D12 resources are missing.");
    }
}

void DMLHelper::WaitForGpuIdle() {
    if (!pImpl || !pImpl->commandQueue || !pImpl->d3d12Device) return;

    ComPtr<ID3D12Fence> fence;
    if (FAILED(pImpl->d3d12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) return;
    
    HANDLE fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (fenceEvent == nullptr) return;

    pImpl->commandQueue->Signal(fence.Get(), 1);
    fence->SetEventOnCompletion(1, fenceEvent);
    WaitForSingleObject(fenceEvent, INFINITE);
    CloseHandle(fenceEvent);
}


ID3D12Device* DMLHelper::GetD3D12Device() const {
    return pImpl ? pImpl->d3d12Device.Get() : nullptr;
}

ID3D12CommandQueue* DMLHelper::GetCommandQueue() const {
    return pImpl ? pImpl->commandQueue.Get() : nullptr;
}

}