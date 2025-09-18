// File: src/VirtuaCast/OnnxRuntime.h
#pragma once

#include <memory>
#include <string>
#include <windows.h>
#include "onnxruntime_cxx_api.h"

struct ID3D11Device;
struct ID3D12Resource;

namespace VirtuaCast {
    
    class Console;
    class DMLHelper;

    class OnnxRuntime {
    public:
        OnnxRuntime();
        ~OnnxRuntime();

        HRESULT Initialize(ID3D11Device* d3d11Device, Console* console);
        void Teardown();

        std::unique_ptr<Ort::Session> CreateSession(const std::wstring& model_path);
        
        Ort::Value CreateTensorFromD3DResource(
            ID3D12Resource* resource,
            const std::vector<int64_t>& shape,
            ONNXTensorElementDataType type
        );
        
        template<typename T>
        Ort::Value CreateCpuTensor(
            const std::vector<T>& data,
            const std::vector<int64_t>& shape
        );

        DMLHelper& GetDMLHelper();
        void WaitForGpuIdle();

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };
}