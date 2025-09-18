// File: src/VirtuaCast/CreateBlob.h

#pragma once

#include <d3d11.h>
#include <wrl/client.h>
#include <memory>

namespace VirtuaCast {

    class BlobCreator {
    public:
        BlobCreator();
        ~BlobCreator();

        HRESULT Initialize(ID3D11Device* device);

        // Executes a compute shader to resize the input SRV and convert it
        // into a planar (NCHW) float blob in the output UAV.
        void Execute(
            ID3D11DeviceContext* context,
            ID3D11ShaderResourceView* input_srv,
            ID3D11UnorderedAccessView* output_uav,
            UINT output_width,
            UINT output_height
        );

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };

} // namespace VirtuaCast