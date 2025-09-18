// File: src/VirtuaCast/BufferManager11.cpp

#include "BufferManager11.h"

using namespace Microsoft::WRL;

namespace VirtuaCast {

struct BufferManager11::Impl {
    ComPtr<ID3D11Device> m_device;
};

BufferManager11::BufferManager11(ID3D11Device* device) : pImpl(std::make_unique<Impl>()) {
    pImpl->m_device = device;
}

BufferManager11::~BufferManager11() = default;

HRESULT BufferManager11::CreateVertexBuffer(UINT byte_width, bool is_dynamic, const void* initial_data, ID3D11Buffer** out_buffer) const {
    D3D11_BUFFER_DESC desc = {};
    desc.ByteWidth = byte_width;
    desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    desc.Usage = is_dynamic ? D3D11_USAGE_DYNAMIC : D3D11_USAGE_DEFAULT;
    desc.CPUAccessFlags = is_dynamic ? D3D11_CPU_ACCESS_WRITE : 0;

    if (initial_data) {
        D3D11_SUBRESOURCE_DATA subData = { initial_data, 0, 0 };
        return pImpl->m_device->CreateBuffer(&desc, &subData, out_buffer);
    }
    return pImpl->m_device->CreateBuffer(&desc, nullptr, out_buffer);
}

HRESULT BufferManager11::CreateIndexBuffer(UINT byte_width, bool is_dynamic, const void* initial_data, ID3D11Buffer** out_buffer) const {
    D3D11_BUFFER_DESC desc = {};
    desc.ByteWidth = byte_width;
    desc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    desc.Usage = is_dynamic ? D3D11_USAGE_DYNAMIC : D3D11_USAGE_DEFAULT;
    desc.CPUAccessFlags = is_dynamic ? D3D11_CPU_ACCESS_WRITE : 0;
    
    if (initial_data) {
        D3D11_SUBRESOURCE_DATA subData = { initial_data, 0, 0 };
        return pImpl->m_device->CreateBuffer(&desc, &subData, out_buffer);
    }
    return pImpl->m_device->CreateBuffer(&desc, nullptr, out_buffer);
}

HRESULT BufferManager11::CreateConstantBuffer(UINT byte_width, ID3D11Buffer** out_buffer) const {
    D3D11_BUFFER_DESC desc = {};
    desc.ByteWidth = byte_width;
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    desc.Usage = D3D11_USAGE_DYNAMIC;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    
    return pImpl->m_device->CreateBuffer(&desc, nullptr, out_buffer);
}


HRESULT BufferManager11::CreateStructuredBuffer(UINT element_count, UINT element_stride, bool is_cpu_writable, bool is_gpu_writable, const void* initial_data, ID3D11Buffer** out_buffer, ID3D11ShaderResourceView** out_srv, ID3D11UnorderedAccessView** out_uav) const {
    D3D11_BUFFER_DESC desc = {};
    desc.ByteWidth = element_count * element_stride;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    desc.StructureByteStride = element_stride;
    desc.BindFlags = 0;
    if (out_srv) desc.BindFlags |= D3D11_BIND_SHADER_RESOURCE;
    if (out_uav) desc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;

    if (is_cpu_writable) {
        desc.Usage = D3D11_USAGE_DYNAMIC;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    } else if (is_gpu_writable) {
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.CPUAccessFlags = 0;
    } else {
        desc.Usage = D3D11_USAGE_IMMUTABLE;
    }

    HRESULT hr;
    if (initial_data) {
        D3D11_SUBRESOURCE_DATA subData = { initial_data, 0, 0 };
        hr = pImpl->m_device->CreateBuffer(&desc, &subData, out_buffer);
    } else {
        hr = pImpl->m_device->CreateBuffer(&desc, nullptr, out_buffer);
    }
    if (FAILED(hr)) return hr;

    if (out_srv) {
        hr = pImpl->m_device->CreateShaderResourceView(*out_buffer, nullptr, out_srv);
        if (FAILED(hr)) return hr;
    }

    if (out_uav) {
        hr = pImpl->m_device->CreateUnorderedAccessView(*out_buffer, nullptr, out_uav);
        if (FAILED(hr)) return hr;
    }

    return S_OK;
}

} // namespace VirtuaCast