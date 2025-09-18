// File: src/VirtuaCast/Generator.cpp

#include "Generator.h"
#include <d3dcompiler.h>
#include <vector>
#include <chrono>

#pragma comment(lib, "d3dcompiler.lib")

using namespace Microsoft::WRL;

namespace {
    const char* g_shaderHLSL = R"(
        cbuffer Constants : register(b0) { float4 bar_rect; float2 resolution; };
        struct PSInput { float4 position : SV_POSITION; };
        float3 SmpteBar75(float u) {
            if      (u < 1.0/7.0) return float3(0.75, 0.75, 0.75);
            else if (u < 2.0/7.0) return float3(0.75, 0.75, 0.00);
            else if (u < 3.0/7.0) return float3(0.00, 0.75, 0.75);
            else if (u < 4.0/7.0) return float3(0.00, 0.75, 0.00);
            else if (u < 5.0/7.0) return float3(0.75, 0.00, 0.75);
            else if (u < 6.0/7.0) return float3(0.75, 0.00, 0.00);
            else                  return float3(0.00, 0.00, 0.75);
        }
        PSInput VSMain(uint id : SV_VertexID) {
            PSInput output; float2 uv = float2((id << 1) & 2, id & 2);
            output.position = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0, 1); return output;
        }
        float4 PSMain(PSInput input) : SV_TARGET {
            float2 ndc = input.position.xy / resolution.xy * float2(2.0, -2.0) + float2(-1.0, 1.0);
            float3 bg = lerp(float3(0.01, 0.01, 0.03), float3(0.03, 0.02, 0.06), input.position.y / resolution.y);
            bg *= (1.0 - 0.55 * smoothstep(0.55, 1.10, length(ndc)));
            float xL = bar_rect.x; float yT = bar_rect.y; float w = bar_rect.z; float h = bar_rect.w;
            if (ndc.x < xL || ndc.x > (xL + w) || ndc.y > yT || ndc.y < (yT - h)) {
                 return float4(bg, 1.0);
            }
            float u = (ndc.x - xL) / w;
            float v = (yT - ndc.y) / h;
            float3 color = float3(0,0,0);
            if (v < (2.0/3.0)) {
                color = SmpteBar75(u);
            }
            else if (v < 0.75) {
                if      (u < 1.0/7.0) color = float3(0.0, 0.0, 0.75);
                else if (u < 2.0/7.0) color = float3(0.0, 0.0, 0.0);
                else if (u < 3.0/7.0) color = float3(0.75, 0.0, 0.75);
                else if (u < 4.0/7.0) color = float3(0.0, 0.0, 0.0);
                else if (u < 5.0/7.0) color = float3(0.0, 0.75, 0.75);
                else if (u < 6.0/7.0) color = float3(0.0, 0.0, 0.0);
                else                  color = float3(0.75, 0.75, 0.75);
            }
            else {
                if (u < 5.0/7.0) {
                    if      (u < 2.5/7.0) color = float3(0.0, 0.0, 0.0);
                    else if (u < 3.5/7.0) color = float3(1.0, 1.0, 1.0);
                    else                  color = float3(0.0, 0.0, 0.0);
                } else {
                    float pluge_u = (u - 5.0/7.0) / (2.0/7.0);
                    if      (pluge_u < 1.0/3.0) color = float3(0.03, 0.03, 0.03);
                    else if (pluge_u < 2.0/3.0) color = float3(0.075, 0.075, 0.075);
                    else                        color = float3(0.115, 0.115, 0.115);
                }
            }
            return float4(color, 1.0);
        }
    )";
}

namespace VirtuaCast {

struct Generator::Impl {
    ComPtr<ID3D11Device> m_device;
    ComPtr<ID3D11Texture2D> m_texture;
    ComPtr<ID3D11ShaderResourceView> m_srv;
    ComPtr<ID3D11RenderTargetView> m_rtv;
    ComPtr<ID3D11VertexShader> m_vertexShader;
    ComPtr<ID3D11PixelShader> m_pixelShader;
    ComPtr<ID3D11Buffer> m_constantBuffer;

    // FIX: Restored this struct to its correct definition, matching the shader.
    struct ConstantBufferData { 
        float bar_rect[4]; 
        float resolution[2]; 
        float padding[2]; 
    };
    
    float gBarPosX = 0.0f, gBarPosY = 0.0f;
    float gBarVelX = 0.2f, gBarVelY = 0.3f;
    std::chrono::steady_clock::time_point gLastFrameTime;

    Impl() : gLastFrameTime(std::chrono::steady_clock::now()) {}
};

Generator::Generator() : pImpl(std::make_unique<Impl>()) {}
Generator::~Generator() = default;

HRESULT Generator::Initialize(ID3D11Device* device) {
    pImpl->m_device = device;
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = 512;
    desc.Height = 512;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    
    HRESULT hr = pImpl->m_device->CreateTexture2D(&desc, nullptr, &pImpl->m_texture);
    if (FAILED(hr)) return hr;
    hr = pImpl->m_device->CreateShaderResourceView(pImpl->m_texture.Get(), nullptr, &pImpl->m_srv);
    if (FAILED(hr)) return hr;
    hr = pImpl->m_device->CreateRenderTargetView(pImpl->m_texture.Get(), nullptr, &pImpl->m_rtv);
    if (FAILED(hr)) return hr;

    ComPtr<ID3DBlob> vsBlob, psBlob, errorBlob;
    hr = D3DCompile(g_shaderHLSL, strlen(g_shaderHLSL), nullptr, nullptr, nullptr, "VSMain", "vs_5_0", 0, 0, &vsBlob, &errorBlob);
    if (FAILED(hr)) return hr;
    hr = D3DCompile(g_shaderHLSL, strlen(g_shaderHLSL), nullptr, nullptr, nullptr, "PSMain", "ps_5_0", 0, 0, &psBlob, &errorBlob);
    if (FAILED(hr)) return hr;

    hr = device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &pImpl->m_vertexShader);
    if (FAILED(hr)) return hr;
    hr = device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &pImpl->m_pixelShader);
    if (FAILED(hr)) return hr;

    D3D11_BUFFER_DESC cbDesc = {};
    cbDesc.ByteWidth = sizeof(Impl::ConstantBufferData);
    cbDesc.Usage = D3D11_USAGE_DYNAMIC;
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    hr = device->CreateBuffer(&cbDesc, nullptr, &pImpl->m_constantBuffer);
    return hr;
}

void Generator::Tick(ID3D11DeviceContext* context) {
    if (!pImpl->m_texture) return;
    
    auto currentTime = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(currentTime - pImpl->gLastFrameTime).count();
    pImpl->gLastFrameTime = currentTime;

    const float RENDER_W = 512.0f;
    const float RENDER_H = 512.0f;
    const float BAR_BASE_WIDTH_NDC = 1.0f;
    const float BAR_ASPECT_RATIO = 16.0f / 10.0f;
    const float BAR_HEIGHT_NDC = BAR_BASE_WIDTH_NDC / BAR_ASPECT_RATIO * (RENDER_W / RENDER_H);

    pImpl->gBarPosX += pImpl->gBarVelX * dt;
    pImpl->gBarPosY += pImpl->gBarVelY * dt;
    if (abs(pImpl->gBarPosX) > (1.0f - BAR_BASE_WIDTH_NDC / 2.f)) {
        pImpl->gBarVelX *= -1.0f;
        pImpl->gBarPosX = (1.0f - BAR_BASE_WIDTH_NDC / 2.f) * (pImpl->gBarPosX > 0 ? 1.f : -1.f);
    }
    if (abs(pImpl->gBarPosY) > (1.0f - BAR_HEIGHT_NDC / 2.f)) {
        pImpl->gBarVelY *= -1.0f;
        pImpl->gBarPosY = (1.0f - BAR_HEIGHT_NDC / 2.f) * (pImpl->gBarPosY > 0 ? 1.f : -1.f);
    }

    D3D11_VIEWPORT vp = { 0, 0, RENDER_W, RENDER_H, 0, 1 };
    context->RSSetViewports(1, &vp);
    context->OMSetRenderTargets(1, pImpl->m_rtv.GetAddressOf(), nullptr);

    D3D11_MAPPED_SUBRESOURCE mapped;
    if (SUCCEEDED(context->Map(pImpl->m_constantBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        Impl::ConstantBufferData cb;
        cb.resolution[0] = RENDER_W; 
        cb.resolution[1] = RENDER_H;
        cb.bar_rect[0] = pImpl->gBarPosX - BAR_BASE_WIDTH_NDC / 2.f; 
        cb.bar_rect[1] = pImpl->gBarPosY + BAR_HEIGHT_NDC / 2.f; 
        cb.bar_rect[2] = BAR_BASE_WIDTH_NDC; 
        cb.bar_rect[3] = BAR_HEIGHT_NDC;
        memcpy(mapped.pData, &cb, sizeof(cb));
        context->Unmap(pImpl->m_constantBuffer.Get(), 0);
    }

    context->VSSetShader(pImpl->m_vertexShader.Get(), nullptr, 0);
    context->PSSetShader(pImpl->m_pixelShader.Get(), nullptr, 0);
    context->PSSetConstantBuffers(0, 1, pImpl->m_constantBuffer.GetAddressOf());
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context->Draw(3, 0);
}

ID3D11ShaderResourceView* Generator::GetShaderResourceView() const {
    return pImpl->m_srv.Get();
}

}