// File: src/VirtuaCast/OnnxRuntime.cpp
#include "OnnxRuntime.h"
#include "DMLHelper.h"
#include <vector>
#include "Console.h"
#include <d3d12.h>
#include <numeric>
#include <functional>

namespace VirtuaCast {

struct OnnxRuntime::Impl {
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::MemoryInfo memory_info{nullptr};
    std::unique_ptr<DMLHelper> dml_helper;
    Console* pConsole = nullptr;

    Impl() 
        : env(ORT_LOGGING_LEVEL_WARNING, "VirtuaCastORT"),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {}
};

OnnxRuntime::OnnxRuntime() : pImpl(std::make_unique<Impl>()) {}
OnnxRuntime::~OnnxRuntime() { Teardown(); }

HRESULT OnnxRuntime::Initialize(ID3D11Device* d3d11Device, Console* console) {
    pImpl->pConsole = console;
    pImpl->dml_helper = std::make_unique<DMLHelper>();
    HRESULT hr = pImpl->dml_helper->Initialize(d3d11Device);
    if (FAILED(hr)) {
        if(pImpl->pConsole) pImpl->pConsole->AddLog("[ERROR] DMLHelper initialization failed. ONNX will use CPU.");
        pImpl->dml_helper.reset();
        return hr;
    }
    try {
        pImpl->dml_helper->ConfigureSessionOptions(pImpl->session_options);
        if(pImpl->pConsole) pImpl->pConsole->AddLog("DirectML execution provider configured successfully.");
    } catch (const Ort::Exception& e) {
        if(pImpl->pConsole) pImpl->pConsole->AddLog("[ERROR] Failed to configure DirectML: %s", e.what());
        pImpl->dml_helper.reset();
    }
    pImpl->session_options.SetIntraOpNumThreads(1);
    pImpl->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return S_OK;
}

void OnnxRuntime::Teardown() { pImpl.reset(); }

std::unique_ptr<Ort::Session> OnnxRuntime::CreateSession(const std::wstring& model_path) {
    return std::make_unique<Ort::Session>(pImpl->env, model_path.c_str(), pImpl->session_options);
}

Ort::Value OnnxRuntime::CreateTensorFromD3DResource(
    ID3D12Resource* resource,
    const std::vector<int64_t>& shape,
    ONNXTensorElementDataType type
) {
    if (!pImpl->dml_helper) {
        throw std::runtime_error("DML Helper not initialized for GPU tensor creation.");
    }
    
    size_t element_count = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    
    return Ort::Value::CreateTensor(pImpl->memory_info, resource, element_count, shape.data(), shape.size(), type);
}

void OnnxRuntime::WaitForGpuIdle() {
    if (pImpl->dml_helper) {
        pImpl->dml_helper->WaitForGpuIdle();
    }
}

template<typename T>
Ort::Value OnnxRuntime::CreateCpuTensor(const std::vector<T>& data, const std::vector<int64_t>& shape) {
    return Ort::Value::CreateTensor<T>(pImpl->memory_info, const_cast<T*>(data.data()), data.size(), shape.data(), shape.size());
}
template Ort::Value OnnxRuntime::CreateCpuTensor<float>(const std::vector<float>&, const std::vector<int64_t>&);

DMLHelper& OnnxRuntime::GetDMLHelper() {
    if (!pImpl->dml_helper) {
        throw std::runtime_error("Attempted to access DMLHelper when it was not initialized.");
    }
    return *pImpl->dml_helper;
}

}