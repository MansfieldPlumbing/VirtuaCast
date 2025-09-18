// File: src/VirtuaCast/BufferManager12.h

#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <memory>
#include <vector>

namespace VirtuaCast {

    class BufferManager12 {
    public:
        BufferManager12(ID3D12Device* device);
        ~BufferManager12();

        HRESULT CreateUploadBuffer(UINT64 byte_size, ID3D12Resource** out_resource, void** out_mapped_data);
        
        HRESULT CreateDefaultBuffer(
            UINT64 byte_size, 
            const void* initial_data,
            ID3D12GraphicsCommandList* command_list, 
            ID3D12Resource** out_resource,
            ID3D12Resource** out_upload_resource // This must be kept alive until the command list executes
        );

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };

} // namespace VirtuaCast