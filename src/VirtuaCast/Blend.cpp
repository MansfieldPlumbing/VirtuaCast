#include "Blend.h"
#include <d3dcompiler.h>

#pragma comment(lib, "d3dcompiler.lib")

using namespace Microsoft::WRL;

namespace {
    const char* g_blendCS = R"(
        struct TransformConstants
        {
            float2x3 inverseTransformMatrix;
        };

        cbuffer PostprocessCB : register(b0) {
            TransformConstants transform;
        };

        #define MODEL_OUTPUT_SIZE 128

        Texture2D<float> g_SwappedFaceTexture : register(t0);
        Texture2D<float> g_BlendingMask : register(t1);
        Texture2D<float4> g_OriginalFrame : register(t2);
        SamplerState g_LinearSampler : register(s0);
        RWTexture2D<float4> g_OutputImage : register(u0);

        [numthreads(8, 8, 1)]
        void main(uint2 DTid : SV_DispatchThreadID)
        {
            uint width, height;
            g_OutputImage.GetDimensions(width, height);
            if (DTid.x >= width || DTid.y >= height) return;

            float2 full_res_uv = (float2)(DTid.xy + 0.5f) / float2(width, height);
            float4 original_pixel = g_OriginalFrame.SampleLevel(g_LinearSampler, full_res_uv, 0);

            float2 swapped_face_coords;
            swapped_face_coords.x = transform.inverseTransformMatrix._11 * DTid.x + transform.inverseTransformMatrix._12 * DTid.y + transform.inverseTransformMatrix._13;
            swapped_face_coords.y = transform.inverseTransformMatrix._21 * DTid.x + transform.inverseTransformMatrix._22 * DTid.y + transform.inverseTransformMatrix._23;
            
            float4 final_pixel = original_pixel;

            if (swapped_face_coords.x >= -0.5f && swapped_face_coords.x < (MODEL_OUTPUT_SIZE - 0.5f) &&
                swapped_face_coords.y >= -0.5f && swapped_face_coords.y < (MODEL_OUTPUT_SIZE - 0.5f))
            {
                float2 swapped_uv = (swapped_face_coords + 0.5f) / float2(MODEL_OUTPUT_SIZE, MODEL_OUTPUT_SIZE * 3.0f);
                
                float r = g_SwappedFaceTexture.SampleLevel(g_LinearSampler, swapped_uv, 0);
                float g = g_SwappedFaceTexture.SampleLevel(g_LinearSampler, swapped_uv + float2(0, 1.0/3.0), 0);
                float b = g_SwappedFaceTexture.SampleLevel(g_LinearSampler, swapped_uv + float2(0, 2.0/3.0), 0);
                
                float3 swapped_bgr = float3(b, g, r);
                
                float2 mask_uv = (swapped_face_coords + 0.5f) / float2(MODEL_OUTPUT_SIZE, MODEL_OUTPUT_SIZE);
                float mask_alpha = g_BlendingMask.SampleLevel(g_LinearSampler, mask_uv, 0).r;

                float3 blended_rgb = lerp(original_pixel.rgb, swapped_bgr, mask_alpha);
                final_pixel = float4(blended_rgb, original_pixel.a);
            }
             g_OutputImage[DTid] = final_pixel;
        }
    )";
}

namespace VirtuaCast {

struct Blend::Impl {
    ComPtr<ID3D11Device> m_device;
    ComPtr<ID3D11ComputeShader> m_computeShader;
    ComPtr<ID3D11Buffer> m_constantBuffer;
    ComPtr<ID3D11SamplerState> m_sampler;
};

Blend::Blend() : pImpl(std::make_unique<Impl>()) {}
Blend::~Blend() = default;

HRESULT Blend::Initialize(ID3D11Device* device) {
    pImpl->m_device = device;
    ComPtr<ID3DBlob> csBlob, errorBlob;
    HRESULT hr = D3DCompile(g_blendCS, strlen(g_blendCS), "BlendCS", nullptr, nullptr, "main", "cs_5_0", 0, 0, &csBlob, &errorBlob);
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
    sampDesc.BorderColor[0] = 0.0f;
    sampDesc.BorderColor[1] = 0.0f;
    sampDesc.BorderColor[2] = 0.0f;
    sampDesc.BorderColor[3] = 0.0f;
    hr = device->CreateSamplerState(&sampDesc, &pImpl->m_sampler);
    return hr;
}

void Blend::Execute(ID3D11DeviceContext* context, const TransformConstants& constants, ID3D11ShaderResourceView* swapped_face_buffer_srv, ID3D11ShaderResourceView* mask_srv, ID3D11ShaderResourceView* original_frame_srv, ID3D11UnorderedAccessView* final_frame_uav, UINT frame_width, UINT frame_height) {
    D3D11_MAPPED_SUBRESOURCE mapped;
    context->Map(pImpl->m_constantBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    memcpy(mapped.pData, &constants, sizeof(constants));
    context->Unmap(pImpl->m_constantBuffer.Get(), 0);

    ID3D11ShaderResourceView* srvs[] = { swapped_face_buffer_srv, mask_srv, original_frame_srv };

    context->CSSetShader(pImpl->m_computeShader.Get(), nullptr, 0);
    context->CSSetConstantBuffers(0, 1, pImpl->m_constantBuffer.GetAddressOf());
    context->CSSetShaderResources(0, 3, srvs);
    context->CSSetSamplers(0, 1, pImpl->m_sampler.GetAddressOf());
    context->CSSetUnorderedAccessViews(0, 1, &final_frame_uav, nullptr);

    context->Dispatch((frame_width + 7) / 8, (frame_height + 7) / 8, 1);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
    context->CSSetUnorderedAccessViews(0, 1, nullUAVs, nullptr);
    ID3D11ShaderResourceView* nullSRVs[] = { nullptr, nullptr, nullptr };
    context->CSSetShaderResources(0, 3, nullSRVs);
}
}