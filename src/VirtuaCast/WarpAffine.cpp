#include "WarpAffine.h"
#include <d3dcompiler.h>

#pragma comment(lib, "d3dcompiler.lib")

using namespace Microsoft::WRL;

namespace {
    const char* g_warpAffineCS = R"(
        struct TransformConstants
        {
            float2x3 inverseTransformMatrix;
        };

        cbuffer PreprocessCB : register(b0) {
            TransformConstants transform;
        };

        #define MODEL_INPUT_SIZE 128

        Texture2D<float4> g_InputFrameBGRA : register(t0);
        SamplerState g_LinearSampler : register(s0);
        RWTexture2D<float> g_ModelInputTexture : register(u0);

        [numthreads(8, 8, 1)]
        void main(uint2 DTid : SV_DispatchThreadID)
        {
            if (DTid.x >= MODEL_INPUT_SIZE || DTid.y >= MODEL_INPUT_SIZE) return;

            float2 warped_coords;
            warped_coords.x = transform.inverseTransformMatrix._11 * DTid.x + transform.inverseTransformMatrix._12 * DTid.y + transform.inverseTransformMatrix._13;
            warped_coords.y = transform.inverseTransformMatrix._21 * DTid.x + transform.inverseTransformMatrix._22 * DTid.y + transform.inverseTransformMatrix._23;

            uint width, height;
            g_InputFrameBGRA.GetDimensions(width, height);
            float2 uv = warped_coords / float2(width, height);

            float4 bgra_pixel = float4(0.0f, 0.0f, 0.0f, 1.0f);
            if (uv.x >= 0.0f && uv.x <= 1.0f && uv.y >= 0.0f && uv.y <= 1.0f) {
                bgra_pixel = g_InputFrameBGRA.SampleLevel(g_LinearSampler, uv, 0);
            }
            
            float r = bgra_pixel.b;
            float g = bgra_pixel.g;
            float b = bgra_pixel.r;

            uint y_r = DTid.y;
            uint y_g = DTid.y + MODEL_INPUT_SIZE;
            uint y_b = DTid.y + MODEL_INPUT_SIZE * 2;

            g_ModelInputTexture[uint2(DTid.x, y_r)] = r;
            g_ModelInputTexture[uint2(DTid.x, y_g)] = g;
            g_ModelInputTexture[uint2(DTid.x, y_b)] = b;
        }
    )";
}

namespace VirtuaCast {

struct WarpAffine::Impl {
    ComPtr<ID3D11Device> m_device;
    ComPtr<ID3D11ComputeShader> m_computeShader;
    ComPtr<ID3D11Buffer> m_constantBuffer;
    ComPtr<ID3D11SamplerState> m_sampler;
};

WarpAffine::WarpAffine() : pImpl(std::make_unique<Impl>()) {}
WarpAffine::~WarpAffine() = default;

HRESULT WarpAffine::Initialize(ID3D11Device* device) {
    pImpl->m_device = device;
    ComPtr<ID3DBlob> csBlob, errorBlob;
    HRESULT hr = D3DCompile(g_warpAffineCS, strlen(g_warpAffineCS), "WarpCS", nullptr, nullptr, "main", "cs_5_0", 0, 0, &csBlob, &errorBlob);
    if (FAILED(hr)) return hr;
    hr = device->CreateComputeShader(csBlob->GetBufferPointer(), csBlob->GetBufferSize(), nullptr, &pImpl->m_computeShader);
    if (FAILED(hr)) return hr;
    D3D11_BUFFER_DESC cbDesc = {};
    cbDesc.ByteWidth = sizeof(TransformConstants);
    cbDesc.Usage = D3D11_USAGE_DYNAMIC;
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    hr = device->CreateBuffer(&cbDesc, nullptr, &pImpl->m_constantBuffer);
    if (FAILED(hr)) return hr;
    D3D11_SAMPLER_DESC sampDesc = {};
    sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_BORDER;
    sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_BORDER;
    sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_BORDER;
    hr = device->CreateSamplerState(&sampDesc, &pImpl->m_sampler);
    return hr;
}

void WarpAffine::Execute(ID3D11DeviceContext* context, const TransformConstants& constants, ID3D11ShaderResourceView* input_srv, ID3D11UnorderedAccessView* output_uav) {
    D3D11_MAPPED_SUBRESOURCE mapped;
    context->Map(pImpl->m_constantBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    memcpy(mapped.pData, &constants, sizeof(constants));
    context->Unmap(pImpl->m_constantBuffer.Get(), 0);

    context->CSSetShader(pImpl->m_computeShader.Get(), nullptr, 0);
    context->CSSetConstantBuffers(0, 1, pImpl->m_constantBuffer.GetAddressOf());
    context->CSSetShaderResources(0, 1, &input_srv);
    context->CSSetSamplers(0, 1, pImpl->m_sampler.GetAddressOf());
    context->CSSetUnorderedAccessViews(0, 1, &output_uav, nullptr);

    context->Dispatch(128 / 8, 128 / 8, 1);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
    context->CSSetUnorderedAccessViews(0, 1, nullUAVs, nullptr);
    ID3D11ShaderResourceView* nullSRVs[] = { nullptr };
    context->CSSetShaderResources(0, 1, nullSRVs);
}
}