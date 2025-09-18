// File: src/VirtuaCast/BufferManager11.h

#pragma once

#include <d3d11.h>
#include <wrl/client.h>
#include <memory>
#include <vector>

namespace VirtuaCast {

    class BufferManager11 {
    public:
        BufferManager11(ID3D11Device* device);
        ~BufferManager11();

        HRESULT CreateVertexBuffer(UINT byte_width, bool is_dynamic, const void* initial_data, ID3D11Buffer** out_buffer) const;
        HRESULT CreateIndexBuffer(UINT byte_width, bool is_dynamic, const void* initial_data, ID3D11Buffer** out_buffer) const;
        HRESULT CreateConstantBuffer(UINT byte_width, ID3D11Buffer** out_buffer) const;
        HRESULT CreateStructuredBuffer(UINT element_count, UINT element_stride, bool is_cpu_writable, bool is_gpu_writable, const void* initial_data, ID3D11Buffer** out_buffer, ID3D11ShaderResourceView** out_srv, ID3D11UnorderedAccessView** out_uav) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };

} // namespace VirtuaCast