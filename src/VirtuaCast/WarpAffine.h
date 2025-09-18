#pragma once

#include <d3d11.h>
#include <wrl/client.h>
#include <memory>
#include "GpuTypes.h"

namespace VirtuaCast {

    class WarpAffine {
    public:
        WarpAffine();
        ~WarpAffine();

        HRESULT Initialize(ID3D11Device* device);
        void Execute(
            ID3D11DeviceContext* context,
            const TransformConstants& constants,
            ID3D11ShaderResourceView* input_srv,
            ID3D11UnorderedAccessView* output_uav
        );

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };

}