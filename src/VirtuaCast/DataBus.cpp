#include "DataBus.h"
#include "CameraShader.h"
#include <stdexcept>

#pragma comment(lib, "d3dcompiler.lib")

using namespace Microsoft::WRL;

namespace VirtuaCast {

struct DataBus::Impl {
    ComPtr<ID3D11Device> m_device;
    ComPtr<ID3D11DeviceContext> m_context;

    ComPtr<ID3D11Texture2D> m_privateTexture;
    ComPtr<ID3D11RenderTargetView> m_privateRTV;
    ComPtr<ID3D11ShaderResourceView> m_privateSRV;

    ComPtr<ID3D11VertexShader> m_stretchVS;
    ComPtr<ID3D11PixelShader> m_stretchPS;
    ComPtr<ID3D11SamplerState> m_linearSampler;
    
    UINT m_width = 0;
    UINT m_height = 0;

    HRESULT CreateResources();
    void ReleaseResources();
};

HRESULT DataBus::Impl::CreateResources() {
    ReleaseResources();

    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = m_width;
    desc.Height = m_height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;

    HRESULT hr = m_device->CreateTexture2D(&desc, nullptr, &m_privateTexture);
    if (FAILED(hr)) return hr;

    hr = m_device->CreateRenderTargetView(m_privateTexture.Get(), nullptr, &m_privateRTV);
    if (FAILED(hr)) return hr;
    
    hr = m_device->CreateShaderResourceView(m_privateTexture.Get(), nullptr, &m_privateSRV);
    return hr;
}

void DataBus::Impl::ReleaseResources() {
    m_privateTexture.Reset();
    m_privateRTV.Reset();
    m_privateSRV.Reset();
}

DataBus::DataBus() : pImpl(std::make_unique<Impl>()) {}
DataBus::~DataBus() { Teardown(); }

HRESULT DataBus::Initialize(ID3D11Device* device, ID3D11DeviceContext* context, UINT width, UINT height) {
    pImpl->m_device = device;
    pImpl->m_context = context;
    pImpl->m_width = width;
    pImpl->m_height = height;

    ComPtr<ID3DBlob> vsBlob, psBlob;
    HRESULT hr = GetLetterboxShader(vsBlob, psBlob);
    if (FAILED(hr)) return hr;

    hr = pImpl->m_device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &pImpl->m_stretchVS);
    if (FAILED(hr)) return hr;
    
    hr = pImpl->m_device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &pImpl->m_stretchPS);
    if (FAILED(hr)) return hr;

    D3D11_SAMPLER_DESC sampDesc = {};
    sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    hr = pImpl->m_device->CreateSamplerState(&sampDesc, &pImpl->m_linearSampler);
    if (FAILED(hr)) return hr;
    
    return pImpl->CreateResources();
}

void DataBus::Teardown() {
    pImpl->ReleaseResources();
    pImpl->m_stretchVS.Reset();
    pImpl->m_stretchPS.Reset();
    pImpl->m_linearSampler.Reset();
}

void DataBus::Resize(UINT newWidth, UINT newHeight) {
    if (pImpl->m_width == newWidth && pImpl->m_height == newHeight) return;
    pImpl->m_width = newWidth;
    pImpl->m_height = newHeight;
    pImpl->CreateResources();
}

ID3D11ShaderResourceView* DataBus::StageAndGetSafeSRV(ID3D11ShaderResourceView* sourceSRV) {
    if (!sourceSRV) return nullptr;

    pImpl->m_context->OMSetRenderTargets(1, pImpl->m_privateRTV.GetAddressOf(), nullptr);
    D3D11_VIEWPORT vp = { 0.0f, 0.0f, (float)pImpl->m_width, (float)pImpl->m_height, 0.0f, 1.0f };
    pImpl->m_context->RSSetViewports(1, &vp);
    pImpl->m_context->IASetInputLayout(nullptr);
    pImpl->m_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    pImpl->m_context->VSSetShader(pImpl->m_stretchVS.Get(), nullptr, 0);
    pImpl->m_context->PSSetShader(pImpl->m_stretchPS.Get(), nullptr, 0);
    pImpl->m_context->PSSetSamplers(0, 1, pImpl->m_linearSampler.GetAddressOf());
    pImpl->m_context->PSSetShaderResources(0, 1, &sourceSRV);
    pImpl->m_context->Draw(3, 0);

    ID3D11RenderTargetView* nullRTV[] = { nullptr };
    pImpl->m_context->OMSetRenderTargets(1, nullRTV, nullptr);
    ID3D11ShaderResourceView* nullSRV[] = { nullptr };
    pImpl->m_context->PSSetShaderResources(0, 1, nullSRV);

    return pImpl->m_privateSRV.Get();
}

ID3D11Texture2D* DataBus::GetInternalTexture() const {
    return pImpl->m_privateTexture.Get();
}

}