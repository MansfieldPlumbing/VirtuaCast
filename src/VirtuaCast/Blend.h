#pragma once

#include <d3d11.h>
#include <wrl/client.h>
#include <memory>
#include "GpuTypes.h"

namespace VirtuaCast {

    class Blend {
    public:
        Blend();
        ~Blend();

        HRESULT Initialize(ID3D11Device* device);
        void Execute(
            ID3D11DeviceContext* context,
            const TransformConstants& constants,
            ID3D11ShaderResourceView* swapped_face_srv,
            ID3D11ShaderResourceView* mask_srv,
            ID3D11ShaderResourceView* original_frame_srv,
            ID3D11UnorderedAccessView* final_frame_uav,
            UINT frame_width,
            UINT frame_height
        );

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };

}