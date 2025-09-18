// A:/Face-On/src/Tegrity/Shaders/shared.h
#ifndef SHARED_H
#define SHARED_H

// This header defines data structures shared between C++ and HLSL.
// It is included by Tegrity.cpp and the HLSL shader files.

#ifdef __cplusplus
    // --- C++ DEFINITIONS ---
    // This code is only seen by the C++ compiler (cl.exe).
    #include <DirectXMath.h> // Provides DirectX Math types where available

    // FIX: Define a custom struct to match HLSL's float2x3 memory layout.
    // DirectXMath does not have a built-in XMFLOAT2X3.
    struct XMFLOAT2X3
    {
        float _11, _12, _13;
        float _21, _22, _23;
    };

    // This C++ struct must have a memory layout identical to the HLSL struct below.
    struct TransformConstants
    {
        // FIX: Use the custom-defined XMFLOAT2X3, not one from the DirectX namespace.
        XMFLOAT2X3 inverseTransformMatrix;
    };

#else
    // --- HLSL DEFINITIONS ---
    // This code is only seen by the HLSL compiler (fxc.exe or dxc.exe).

    // This HLSL struct must have a memory layout identical to the C++ struct above.
    struct TransformConstants
    {
        float2x3 inverseTransformMatrix;
    };

    // NEW: Added HLSL-compatible definition for detected face data, matching the C++ TegrityDetectedFace struct.
    // This is used by the visualization shaders.
    struct DetectedFace
    {
        float4 bbox;      // x1, y1, x2, y2
        float2 kps[5];    // 5 keypoints
        float  det_score;
        float  _padding;  // Ensures 16-byte alignment
    };

#endif // __cplusplus

#endif // SHARED_H