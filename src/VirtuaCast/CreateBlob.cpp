// File: src/VirtuaCast/CreateBlob.cpp

#include "CreateBlob.h"
#include <d3dcompiler.h>

#pragma comment(lib, "d3dcompiler.lib")

using namespace Microsoft::WRL;

namespace {
    // This compute shader performs a resize, color conversion, normalization,
    // and layout transform (Interleaved -> Planar NCHW) in a single GPU pass.
    const char* g_createBlobCS = R"(
        Texture2D<float4>        g_InputTexture    : register(t0);
        RWBuffer<float>          g_OutputBlob      : register(u0);
        SamplerState             g_LinearSampler   : register(s0);

        cbuffer DimensionsCB : register(b0)
        {
            uint2 g_OutputDimensions;
        };

        [numthreads(8, 8, 1)]
        void main(uint2 DTid : SV_DispatchThreadID)
        {
            if (DTid.x >= g_OutputDimensions.x || DTid.y >= g_OutputDimensions.y)
            {
                return;
            }

            // 1. RESIZE: Calculate UVs to sample the source texture.
            // This effectively performs a stretch-blit.
            float2 uv = (float2(DTid.x, DTid.y) + 0.5f) / float2(g_OutputDimensions.x, g_OutputDimensions.y);

            // Sample the source texture using the linear sampler for interpolation.
            float4 pixelBGRA = g_InputTexture.SampleLevel(g_LinearSampler, uv, 0);

            // 2. NORMALIZE: Convert from [0, 255] BGRA to [-1, 1] BGR floats.
            float b = (pixelBGRA.r * 255.0f - 127.5f) * (1.0f / 128.0f);
            float g = (pixelBGRA.g * 255.0f - 127.5f) * (1.0f / 128.0f);
            float r = (pixelBGRA.b * 255.0f - 127.5f) * (1.0f / 128.0f);

            // 3. TRANSFORM LAYOUT: Calculate indices for NCHW (Planar) format.
            uint pixel_index = DTid.y * g_OutputDimensions.x + DTid.x;
            uint num_pixels = g_OutputDimensions.x * g_OutputDimensions.y;
            
            // Channel 0: Blue
            // Channel 1: Green
            // Channel 2: Red
            g_OutputBlob[pixel_index] = b;
            g_OutputBlob[pixel_index + num_pixels] = g;
            g_OutputBlob[pixel_index + 2 * num_pixels] = r;
        }
    )";
}

namespace VirtuaCast {

struct BlobCreator::Impl {
    ComPtr<ID3D11Device> m_device;
    ComPtr<ID3D11ComputeShader> m_computeShader;
    ComPtr<ID3D11Buffer> m_constantBuffer;
    ComPtr<ID3D11SamplerState> m_sampler;
};

BlobCreator::BlobCreator() : pImpl(std::make_unique<Impl>()) {}
BlobCreator::~BlobCreator() = default;

HRESULT BlobCreator::Initialize(ID3D11Device* device) {
    pImpl->m_device = device;
    ComPtr<ID3DBlob> csBlob, errorBlob;
    HRESULT hr = D3DCompile(g_createBlobCS, strlen(g_createBlobCS), "CreateBlobCS", nullptr, nullptr, "main", "cs_5_0", 0, 0, &csBlob, &errorBlob);
    if (FAILED(hr)) return hr;
    hr = device->CreateComputeShader(csBlob->GetBufferPointer(), csBlob->GetBufferSize(), nullptr, &pImpl->m_computeShader);
    if (FAILED(hr)) return hr;

    D3D11_BUFFER_DESC cbDesc = {};
    cbDesc.ByteWidth = 16; // Must be a multiple of 16
    cbDesc.Usage = D3D11_USAGE_DYNAMIC;
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    hr = device->CreateBuffer(&cbDesc, nullptr, &pImpl->m_constantBuffer);
    if (FAILED(hr)) return hr;

    D3D11_SAMPLER_DESC sampDesc = {};
    sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_BORDER; // Use border color for out-of-bounds samples
    sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_BORDER;
    sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_BORDER;
    sampDesc.BorderColor[0] = sampDesc.BorderColor[1] = sampDesc.BorderColor[2] = sampDesc.BorderColor[3] = 0;
    hr = device->CreateSamplerState(&sampDesc, &pImpl->m_sampler);
    return hr;
}

void BlobCreator::Execute(
    ID3D11DeviceContext* context,
    ID3D11ShaderResourceView* input_srv,
    ID3D11UnorderedAccessView* output_uav,
    UINT output_width,
    UINT output_height
) {
    // Update constant buffer
    D3D11_MAPPED_SUBRESOURCE mapped;
    context->Map(pImpl->m_constantBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    UINT dims[] = { output_width, output_height };
    memcpy(mapped.pData, dims, sizeof(dims));
    context->Unmap(pImpl->m_constantBuffer.Get(), 0);

    // Set pipeline state
    context->CSSetShader(pImpl->m_computeShader.Get(), nullptr, 0);
    context->CSSetConstantBuffers(0, 1, pImpl->m_constantBuffer.GetAddressOf());
    context->CSSetShaderResources(0, 1, &input_srv);
    context->CSSetSamplers(0, 1, pImpl->m_sampler.GetAddressOf());
    context->CSSetUnorderedAccessViews(0, 1, &output_uav, nullptr);

    // Dispatch
    context->Dispatch((output_width + 7) / 8, (output_height + 7) / 8, 1);

    // Unbind resources
    ID3D11UnorderedAccessView* nullUAV[] = { nullptr };
    context->CSSetUnorderedAccessViews(0, 1, nullUAV, nullptr);
    ID3D11ShaderResourceView* nullSRV[] = { nullptr };
    context->CSSetShaderResources(0, 1, nullSRV);
}
}