// File: src/VirtuaCast/Producer.h

#pragma once

#include <d3d11_4.h>
#include <wrl/client.h>
#include <memory>
#include <string>

struct ID3D11DeviceContext;
struct ID3D11Texture2D;

namespace VirtuaCast {

    class Producer {
    public:
        Producer();
        ~Producer();

        HRESULT Initialize(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11Texture2D* sharedTexture);
        void Teardown();
        void PublishFrame();

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };

}