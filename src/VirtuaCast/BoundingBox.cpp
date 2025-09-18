// File: src/VirtuaCast/BoundingBox.cpp

#include "BoundingBox.h"
#include <d3dcompiler.h>
#include <vector>
#include <cmath>
#include <algorithm> // For std::min

#pragma comment(lib, "d3dcompiler.lib")

using namespace Microsoft::WRL;

namespace {
    // This shader is trivial. It expects coordinates that are ALREADY in Normalized Device Coordinates.
    const char* g_simpleShaderHLSL = R"(
        struct VS_INPUT {
            float2 pos   : POSITION;
        };
        float4 VSMain(VS_INPUT input) : SV_POSITION {
            return float4(input.pos, 0.5, 1.0);
        }
        float4 PSMain() : SV_TARGET {
            return float4(0.1, 0.55, 1.0, 1.0); // Solid Blue
        }
    )";
    
    // A manually defined constant for pi to avoid platform-specific issues with M_PI.
    const float PI = 3.14159265358979323846f;

    // Define corner drawing parameters in one place.
    const int NUM_CORNER_SEGMENTS = 10; // Increased for smoother corners
    const int MAX_FACES = 10; // The maximum number of faces we can draw
    const size_t VERTICES_PER_BOX = 6 + (4 * NUM_CORNER_SEGMENTS);
}

namespace VirtuaCast {

struct BoundingBoxRenderer::Impl {
    ComPtr<ID3D11Device> m_device;
    ComPtr<ID3D11VertexShader> m_vertexShader;
    ComPtr<ID3D11PixelShader> m_pixelShader;
    ComPtr<ID3D11InputLayout> m_inputLayout;
    ComPtr<ID3D11Buffer> m_vertexBuffer;

    struct Vertex {
        float x, y; // Stores NDC coordinates directly
    };
};

BoundingBoxRenderer::BoundingBoxRenderer() : pImpl(std::make_unique<Impl>()) {}
BoundingBoxRenderer::~BoundingBoxRenderer() { Teardown(); }

HRESULT BoundingBoxRenderer::Initialize(ID3D11Device* device) {
    pImpl->m_device = device;

    ComPtr<ID3DBlob> vsBlob, psBlob, errorBlob;
    HRESULT hr = D3DCompile(g_simpleShaderHLSL, strlen(g_simpleShaderHLSL), nullptr, nullptr, nullptr, "VSMain", "vs_5_0", 0, 0, &vsBlob, &errorBlob);
    if (FAILED(hr)) return hr;
    hr = D3DCompile(g_simpleShaderHLSL, strlen(g_simpleShaderHLSL), nullptr, nullptr, nullptr, "PSMain", "ps_5_0", 0, 0, &psBlob, &errorBlob);
    if (FAILED(hr)) return hr;

    hr = device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &pImpl->m_vertexShader);
    if (FAILED(hr)) return hr;
    hr = device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &pImpl->m_pixelShader);
    if (FAILED(hr)) return hr;

    D3D11_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    hr = device->CreateInputLayout(layout, ARRAYSIZE(layout), vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &pImpl->m_inputLayout);
    if (FAILED(hr)) return hr;
    
    // Create a vertex buffer with a size calculated from our constants.
    D3D11_BUFFER_DESC vbDesc = {};
    vbDesc.ByteWidth = sizeof(Impl::Vertex) * VERTICES_PER_BOX * MAX_FACES;
    vbDesc.Usage = D3D11_USAGE_DYNAMIC;
    vbDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    vbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    hr = device->CreateBuffer(&vbDesc, nullptr, &pImpl->m_vertexBuffer);

    return hr;
}

void BoundingBoxRenderer::Teardown() {
    pImpl->m_vertexBuffer.Reset();
    pImpl->m_inputLayout.Reset();
    pImpl->m_pixelShader.Reset();
    pImpl->m_vertexShader.Reset();
    pImpl->m_device.Reset();
}

void BoundingBoxRenderer::Render(ID3D11DeviceContext* context, const FrameData& frameData) {
    if (frameData.faces.empty() || !pImpl->m_vertexBuffer) {
        return;
    }

    ComPtr<ID3D11Resource> resource;
    frameData.pOutputRTV->GetResource(&resource);
    ComPtr<ID3D11Texture2D> texture;
    resource.As(&texture);
    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);
    
    const float frameWidth = static_cast<float>(desc.Width);
    const float frameHeight = static_cast<float>(desc.Height);

    std::vector<Impl::Vertex> allVertices;
    allVertices.reserve(frameData.faces.size() * VERTICES_PER_BOX);

    const float cornerRadiusPx = 30.0f; // Corner radius in pixels

    for (const auto& face : frameData.faces) {
        float x1_px = face.bbox[0];
        float y1_px = face.bbox[1];
        float x2_px = face.bbox[2];
        float y2_px = face.bbox[3];

        // Ensure the radius isn't larger than half the box size, which would cause artifacts.
        float safeRadius = std::min({cornerRadiusPx, (x2_px - x1_px) / 2.0f, (y2_px - y1_px) / 2.0f});

        // If the box is too small, just draw a simple rectangle
        if (safeRadius <= 0) {
            allVertices.push_back({(x1_px / frameWidth) * 2.0f - 1.0f, (y1_px / frameHeight) * -2.0f + 1.0f});
            allVertices.push_back({(x2_px / frameWidth) * 2.0f - 1.0f, (y1_px / frameHeight) * -2.0f + 1.0f});
            allVertices.push_back({(x2_px / frameWidth) * 2.0f - 1.0f, (y2_px / frameHeight) * -2.0f + 1.0f});
            allVertices.push_back({(x1_px / frameWidth) * 2.0f - 1.0f, (y2_px / frameHeight) * -2.0f + 1.0f});
            allVertices.push_back({(x1_px / frameWidth) * 2.0f - 1.0f, (y1_px / frameHeight) * -2.0f + 1.0f});
            // Fill remaining vertices with duplicates to maintain consistent stride
            for(size_t i = 5; i < VERTICES_PER_BOX; ++i) {
                allVertices.push_back({(x1_px / frameWidth) * 2.0f - 1.0f, (y1_px / frameHeight) * -2.0f + 1.0f});
            }
            continue;
        }
        
        // --- Generate vertices for a continuous LINESTRIP ---
        
        // Start of top line
        allVertices.push_back({(x1_px + safeRadius) / frameWidth * 2.0f - 1.0f, (y1_px) / frameHeight * -2.0f + 1.0f});
        // End of top line
        allVertices.push_back({(x2_px - safeRadius) / frameWidth * 2.0f - 1.0f, (y1_px) / frameHeight * -2.0f + 1.0f});
        
        // Top-right corner
        float centerX = x2_px - safeRadius;
        float centerY = y1_px + safeRadius;
        for (int i = 0; i <= NUM_CORNER_SEGMENTS; ++i) {
            float angle = static_cast<float>(i) / NUM_CORNER_SEGMENTS * (PI / 2.0f);
            float x = centerX + sin(angle) * safeRadius;
            float y = centerY - cos(angle) * safeRadius;
            allVertices.push_back({x / frameWidth * 2.0f - 1.0f, y / frameHeight * -2.0f + 1.0f});
        }
        
        // End of right line
        allVertices.push_back({(x2_px) / frameWidth * 2.0f - 1.0f, (y2_px - safeRadius) / frameHeight * -2.0f + 1.0f});
        
        // Bottom-right corner
        centerX = x2_px - safeRadius;
        centerY = y2_px - safeRadius;
        for (int i = 0; i <= NUM_CORNER_SEGMENTS; ++i) {
            float angle = static_cast<float>(i) / NUM_CORNER_SEGMENTS * (PI / 2.0f);
            float x = centerX + cos(angle) * safeRadius;
            float y = centerY + sin(angle) * safeRadius;
            allVertices.push_back({x / frameWidth * 2.0f - 1.0f, y / frameHeight * -2.0f + 1.0f});
        }
        
        // End of bottom line
        allVertices.push_back({(x1_px + safeRadius) / frameWidth * 2.0f - 1.0f, (y2_px) / frameHeight * -2.0f + 1.0f});
        
        // Bottom-left corner
        centerX = x1_px + safeRadius;
        centerY = y2_px - safeRadius;
        for (int i = 0; i <= NUM_CORNER_SEGMENTS; ++i) {
            float angle = static_cast<float>(i) / NUM_CORNER_SEGMENTS * (PI / 2.0f);
            float x = centerX - sin(angle) * safeRadius;
            float y = centerY + cos(angle) * safeRadius;
            allVertices.push_back({x / frameWidth * 2.0f - 1.0f, y / frameHeight * -2.0f + 1.0f});
        }

        // End of left line
        allVertices.push_back({(x1_px) / frameWidth * 2.0f - 1.0f, (y1_px + safeRadius) / frameHeight * -2.0f + 1.0f});

        // Top-left corner
        centerX = x1_px + safeRadius;
        centerY = y1_px + safeRadius;
        for (int i = 0; i <= NUM_CORNER_SEGMENTS; ++i) {
            float angle = static_cast<float>(i) / NUM_CORNER_SEGMENTS * (PI / 2.0f);
            float x = centerX - cos(angle) * safeRadius;
            float y = centerY - sin(angle) * safeRadius;
            allVertices.push_back({x / frameWidth * 2.0f - 1.0f, y / frameHeight * -2.0f + 1.0f});
        }

        // The line strip is implicitly closed by the logic above. No extra point is needed.
    }

    // Update the vertex buffer with all box vertices at once
    D3D11_MAPPED_SUBRESOURCE mappedVB;
    if (SUCCEEDED(context->Map(pImpl->m_vertexBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedVB))) {
        memcpy(mappedVB.pData, allVertices.data(), allVertices.size() * sizeof(Impl::Vertex));
        context->Unmap(pImpl->m_vertexBuffer.Get(), 0);
    }

    // Set up the rendering pipeline state
    context->OMSetRenderTargets(1, &frameData.pOutputRTV, nullptr);
    context->IASetInputLayout(pImpl->m_inputLayout.Get());
    context->VSSetShader(pImpl->m_vertexShader.Get(), nullptr, 0);
    context->PSSetShader(pImpl->m_pixelShader.Get(), nullptr, 0);
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP);

    UINT stride = sizeof(Impl::Vertex);
    UINT offset = 0;
    context->IASetVertexBuffers(0, 1, pImpl->m_vertexBuffer.GetAddressOf(), &stride, &offset);

    // Draw all boxes, one after the other.
    for (size_t i = 0; i < frameData.faces.size(); ++i) {
        // Use the pre-calculated and CORRECT vertex count for each box.
        context->Draw(VERTICES_PER_BOX, i * VERTICES_PER_BOX);
    }
}

} // namespace VirtuaCast