// File: src/VirtuaCast/FaceEmbedding.cpp

#include "FaceEmbedding.h"
#include "OnnxRuntime.h"
#include "Files.h"
#include "Algorithms.h"
#include "Types.h"
#include "Console.h"
#include <d3d11.h>
#include <wrl/client.h>
#include <vector>

using namespace Microsoft::WRL;
using namespace VirtuaCast;

namespace {
    // Local helper function
    HRESULT GetCpuImageFromGpuTexture(
        ID3D11Device* pDevice,
        ID3D11DeviceContext* pContext,
        ID3D11ShaderResourceView* pSrv,
        std::vector<unsigned char>& out_cpu_data,
        WinTegrity::TegrityImageBuffer& out_buffer_view)
    {
        if (!pDevice || !pContext || !pSrv) return E_INVALIDARG;
        ComPtr<ID3D11Resource> resource;
        pSrv->GetResource(&resource);
        ComPtr<ID3D11Texture2D> texture;
        HRESULT hr = resource.As(&texture);
        if (FAILED(hr)) return hr;
        D3D11_TEXTURE2D_DESC desc;
        texture->GetDesc(&desc);
        D3D11_TEXTURE2D_DESC stagingDesc = desc;
        stagingDesc.Usage = D3D11_USAGE_STAGING;
        stagingDesc.BindFlags = 0;
        stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        stagingDesc.MiscFlags = 0;
        ComPtr<ID3D11Texture2D> stagingTexture;
        hr = pDevice->CreateTexture2D(&stagingDesc, nullptr, &stagingTexture);
        if (FAILED(hr)) return hr;
        pContext->CopyResource(stagingTexture.Get(), texture.Get());
        D3D11_MAPPED_SUBRESOURCE mappedResource;
        hr = pContext->Map(stagingTexture.Get(), 0, D3D11_MAP_READ, 0, &mappedResource);
        if (FAILED(hr)) return hr;
        const UINT bytesPerPixel = 4;
        out_cpu_data.resize(static_cast<size_t>(desc.Width) * desc.Height * bytesPerPixel);
        unsigned char* pSrc = static_cast<unsigned char*>(mappedResource.pData);
        unsigned char* pDst = out_cpu_data.data();
        for (UINT y = 0; y < desc.Height; ++y) {
            memcpy(pDst, pSrc, static_cast<size_t>(desc.Width) * bytesPerPixel);
            pDst += desc.Width * bytesPerPixel;
            pSrc += mappedResource.RowPitch;
        }
        pContext->Unmap(stagingTexture.Get(), 0);
        out_buffer_view.data = out_cpu_data.data();
        out_buffer_view.width = desc.Width;
        out_buffer_view.height = desc.Height;
        out_buffer_view.channels = bytesPerPixel;
        return S_OK;
    }

    void CreateBlobFromBGR(const WinTegrity::TegrityImageBuffer& image, std::vector<float>& out_blob, float scale, float mean_b, float mean_g, float mean_r) {
        const int w = image.width;
        const int h = image.height;
        const size_t num_pixels = static_cast<size_t>(w) * h;
        out_blob.resize(num_pixels * 3);
        
        // --- FIX: Pointers now correctly represent an RGB layout ---
        float* blob_r_plane = out_blob.data();
        float* blob_g_plane = out_blob.data() + num_pixels;
        float* blob_b_plane = out_blob.data() + 2 * num_pixels;
        
        #pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i) {
            // --- FIX: Swap R and B channels to match groundtruth.py (swapRB=True) ---
            // Read from BGR source
            const unsigned char* p = image.data + (static_cast<size_t>(i) * 3);
            const float b_val = static_cast<float>(p[0]);
            const float g_val = static_cast<float>(p[1]);
            const float r_val = static_cast<float>(p[2]);

            // Write to planar RGB blob
            blob_r_plane[i] = (r_val - mean_b) * scale;
            blob_g_plane[i] = (g_val - mean_g) * scale;
            blob_b_plane[i] = (b_val - mean_r) * scale;
        }
    }

}

struct FaceEmbedder::Impl {
    std::unique_ptr<Ort::Session> rec_session;
    Ort::MemoryInfo memory_info{nullptr};
    std::vector<uint8_t> frame_cpu_buffer_rgba;
    std::vector<uint8_t> bgr_buffer;
    std::vector<uint8_t> aligned_face_buffer;
    std::vector<float> blob;
    Impl() : memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}
};

FaceEmbedder::FaceEmbedder() : pImpl(std::make_unique<Impl>()) {}
FaceEmbedder::~FaceEmbedder() = default;

HRESULT FaceEmbedder::Initialize(OnnxRuntime& onnx, const std::string& model_dir, Console* console) {
    try {
        std::wstring model_path = WinTegrity::ToWide(model_dir + "/w600k_r50.onnx");
        pImpl->rec_session = onnx.CreateSession(model_path);
        pImpl->aligned_face_buffer.resize(112 * 112 * 3);
    } catch (const std::exception& e) {
        if (console) console->AddLog("[ERROR] FaceEmbedder failed to load model: %s", e.what());
        OutputDebugStringA(e.what());
        return E_FAIL;
    }
    return S_OK;
}

void FaceEmbedder::Process(FrameData& frameData) {
    if (!pImpl->rec_session || frameData.faces.empty() || !frameData.pInputSRV) {
        return;
    }

    WinTegrity::TegrityImageBuffer cpu_frame_rgba;
    HRESULT hr = GetCpuImageFromGpuTexture(frameData.pDevice, frameData.pContext, frameData.pInputSRV, pImpl->frame_cpu_buffer_rgba, cpu_frame_rgba);
    if (FAILED(hr)) return;

    pImpl->bgr_buffer.resize(static_cast<size_t>(cpu_frame_rgba.width) * cpu_frame_rgba.height * 3);
    WinTegrity::TegrityImageBuffer bgr_img = { pImpl->bgr_buffer.data(), cpu_frame_rgba.width, cpu_frame_rgba.height, 3 };
    #pragma omp parallel for
    for (int i = 0; i < bgr_img.width * bgr_img.height; ++i) {
        memcpy(bgr_img.data + (size_t)i * 3, cpu_frame_rgba.data + (size_t)i * 4, 3);
    }

    for (auto& face : frameData.faces) {
        const int align_size = 112;
        WinTegrity::TegrityImageBuffer aligned_img_dst = { pImpl->aligned_face_buffer.data(), align_size, align_size, 3 };
        
        Eigen::Matrix<double, 5, 2> arcface_dst_pts_112;
        arcface_dst_pts_112 << 38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366, 41.5493, 92.3655, 70.7299, 92.2041;
        
        Eigen::Matrix<double, 2, 3> M = WinTegrity::estimateSimilarityTransform(face.landmarks, arcface_dst_pts_112);
        WinTegrity::warpAffineBilinear(bgr_img, aligned_img_dst, M, nullptr);

        CreateBlobFromBGR(aligned_img_dst, pImpl->blob, 1.f / 127.5f, 127.5f, 127.5f, 127.5f);

        std::vector<int64_t> input_shape = { 1, 3, align_size, align_size };
        auto input_tensor = Ort::Value::CreateTensor<float>(pImpl->memory_info, pImpl->blob.data(), pImpl->blob.size(), input_shape.data(), input_shape.size());
        
        const char* input_names[] = { "input.1" };
        const char* output_names[] = { "683" };
        
        try {
            auto output_tensors = pImpl->rec_session->Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);
            float* embedding_data = output_tensors[0].GetTensorMutableData<float>();
            WinTegrity::normalize_l2(embedding_data, 512);
            face.embedding.resize(512);
            memcpy(face.embedding.data(), embedding_data, 512 * sizeof(float));
        } catch (const Ort::Exception& e) {
            OutputDebugStringA(e.what());
        }
    }
}