#pragma once

#include <d3d11_4.h>
#include <wrl/client.h>
#include <memory>

namespace VirtuaCast {

    class DataBus {
    public:
        DataBus();
        ~DataBus();

        HRESULT Initialize(ID3D11Device* device, ID3D11DeviceContext* context, UINT width, UINT height);
        void Teardown();
        void Resize(UINT newWidth, UINT newHeight);
        ID3D11ShaderResourceView* StageAndGetSafeSRV(ID3D11ShaderResourceView* sourceSRV);
        ID3D11Texture2D* GetInternalTexture() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };

}