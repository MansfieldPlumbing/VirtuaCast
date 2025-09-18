// File: src/VirtuaCast/Generator.h

#pragma once

#include <d3d11.h>
#include <wrl/client.h>
#include <memory>

struct ID3D11DeviceContext;

namespace VirtuaCast {

    class Generator {
    public:
        Generator();
        ~Generator();

        HRESULT Initialize(ID3D11Device* device);
        void Tick(ID3D11DeviceContext* context);
        ID3D11ShaderResourceView* GetShaderResourceView() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };

}