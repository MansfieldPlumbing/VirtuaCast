#pragma once

#include <DirectXMath.h>

namespace VirtuaCast {

    // DirectXMath doesn't have a native XMFLOAT2X3. We define one that matches
    // the memory layout HLSL expects for a float2x3.
    struct XMFLOAT2X3
    {
        float _11, _12, _13;
        float _21, _22, _23;
    };

    struct alignas(16) TransformConstants
    {
        XMFLOAT2X3 inverseTransformMatrix;
        float _padding[2]; 
    };

}