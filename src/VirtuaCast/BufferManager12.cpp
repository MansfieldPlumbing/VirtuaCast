// File: src/VirtuaCast/BufferManager12.cpp

#include "BufferManager12.h"
#include "directx/d3dx12.h"

using namespace Microsoft::WRL;

namespace VirtuaCast {

struct BufferManager12::Impl {
    ComPtr<ID3D12Device> m_device;
};

BufferManager12::BufferManager12(ID3D12Device* device) : pImpl(std::make_unique<Impl>()) {
    pImpl->m_device = device;
}

BufferManager12::~BufferManager12() = default;

HRESULT BufferManager12::CreateUploadBuffer(UINT64 byte_size, ID3D12Resource** out_resource, void** out_mapped_data) {
    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(byte_size);

    HRESULT hr = pImpl->m_device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(out_resource)
    );

    if (SUCCEEDED(hr) && out_mapped_data) {
        (*out_resource)->Map(0, nullptr, out_mapped_data);
    }
    
    return hr;
}

HRESULT BufferManager12::CreateDefaultBuffer(
    UINT64 byte_size, 
    const void* initial_data,
    ID3D12GraphicsCommandList* command_list, 
    ID3D12Resource** out_resource,
    ID3D12Resource** out_upload_resource
) {
    if (!initial_data || !command_list) return E_INVALIDARG;

    auto defaultHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(byte_size);

    // Create the final default buffer resource.
    HRESULT hr = pImpl->m_device->CreateCommittedResource(
        &defaultHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, // Start in copy-destination state
        nullptr,
        IID_PPV_ARGS(out_resource)
    );
    if (FAILED(hr)) return hr;

    // Create a temporary upload heap to transfer the data.
    hr = CreateUploadBuffer(byte_size, out_upload_resource, nullptr);
    if (FAILED(hr)) return hr;

    // Describe the data to be copied.
    D3D12_SUBRESOURCE_DATA subresourceData = {};
    subresourceData.pData = initial_data;
    subresourceData.RowPitch = byte_size;
    subresourceData.SlicePitch = subresourceData.RowPitch;

    // Schedule the copy command. The transition to a final state (e.g., VERTEX_AND_CONSTANT_BUFFER)
    // should happen after this copy completes.
    UpdateSubresources<1>(command_list, *out_resource, *out_upload_resource, 0, 0, 1, &subresourceData);
    
    // NOTE: The caller is responsible for transitioning the resource from D3D12_RESOURCE_STATE_COPY_DEST
    // to its final state after this function returns and before it's used on the GPU.
    // Example: commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(mDefaultBuffer.Get(),
    // D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER));

    return S_OK;
}


} // namespace VirtuaCast