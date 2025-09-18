// File: src/VirtuaCast/BoundingBox.h

#pragma once

#include "Framework.h"
#include <memory>

struct ID3D11Device;
struct ID3D11DeviceContext;

namespace VirtuaCast {

    class BoundingBoxRenderer {
    public:
        BoundingBoxRenderer();
        ~BoundingBoxRenderer();

        HRESULT Initialize(ID3D11Device* device);
        void Teardown();
        void Render(ID3D11DeviceContext* context, const FrameData& frameData);

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };

} // namespace VirtuaCast