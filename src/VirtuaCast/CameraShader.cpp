// File: src/VirtuaCast/CameraShader.cpp

#include "CameraShader.h"
#include <d3dcompiler.h>
#include <stdexcept>

#pragma comment(lib, "d3dcompiler.lib")

namespace VirtuaCast {

// This HLSL code has been simplified to perform a direct stretch-blit.
// It assumes the input texture will be sampled to fill the entire output quad.
static const char* g_stretchShaderHLSL = R"(
    Texture2D<float4> g_textureRGB    : register(t0);
    SamplerState      g_linearSampler : register(s0);

    struct VS_OUTPUT
    {
        float4 position : SV_POSITION;
        float2 uv : TEXCOORD;
    };

    VS_OUTPUT VSMain(uint vid : SV_VertexID)
    {
        VS_OUTPUT output;
        // Creates a full-screen triangle.
        float2 uv = float2((vid << 1) & 2, vid & 2);
        output.position = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
        output.uv = uv; // The UVs for a simple quad are direct.
        return output;
    }

    float4 PSMain(VS_OUTPUT input) : SV_TARGET
    {
        // Direct sampling handles the stretch automatically when rendering a full quad.
        return g_textureRGB.Sample(g_linearSampler, input.uv);
    }
)";

HRESULT GetLetterboxShader(
    Microsoft::WRL::ComPtr<ID3DBlob>& outVertexShader,
    Microsoft::WRL::ComPtr<ID3DBlob>& outPixelShader
) {
    Microsoft::WRL::ComPtr<ID3DBlob> errorBlob;
    HRESULT hr = D3DCompile(
        g_stretchShaderHLSL,
        strlen(g_stretchShaderHLSL),
        nullptr, nullptr, nullptr,
        "VSMain", "vs_5_0",
        0, 0,
        &outVertexShader,
        &errorBlob
    );
    if (FAILED(hr)) return hr;

    hr = D3DCompile(
        g_stretchShaderHLSL,
        strlen(g_stretchShaderHLSL),
        nullptr, nullptr, nullptr,
        "PSMain", "ps_5_0",
        0, 0,
        &outPixelShader,
        &errorBlob
    );
    return hr;
}

} // namespace VirtuaCast