// File: src/VirtuaCast/FaceDetection.cpp

#include "FaceDetection.h"
#include "OnnxRuntime.h"
#include "Files.h"
#include "Algorithms.h"
#include "Types.h"
#include "Console.h"
#include <d3d11.h>
#include <wrl/client.h>
#include <vector>
#include <algorithm>
#include <limits>

using namespace Microsoft::WRL;
using namespace VirtuaCast;

namespace {
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
        
        // Pointers to the start of each color plane in the final RGB blob
        float* blob_r_plane = out_blob.data();
        float* blob_g_plane = out_blob.data() + num_pixels;
        float* blob_b_plane = out_blob.data() + 2 * num_pixels;
        
        #pragma omp parallel for
        for (int i = 0; i < (int)num_pixels; ++i) {
            // Read from BGR source image data
            const unsigned char* p = image.data + (static_cast<size_t>(i) * 3);
            const float b = static_cast<float>(p[0]);
            const float g = static_cast<float>(p[1]);
            const float r = static_cast<float>(p[2]);

            // Write to planar RGB blob
            blob_r_plane[i] = (r - mean_r) * scale;
            blob_g_plane[i] = (g - mean_g) * scale;
            blob_b_plane[i] = (b - mean_b) * scale;
        }
    }

    void NMS(std::vector<FaceResult>& proposals, float iou_threshold, std::vector<FaceResult>& final_faces) {
        final_faces.clear();
        if (proposals.empty()) return;
        std::sort(proposals.begin(), proposals.end(), [](const auto& a, const auto& b) {
            return a.detection_score > b.detection_score;
        });
        std::vector<bool> is_suppressed(proposals.size(), false);
        for (size_t i = 0; i < proposals.size(); ++i) {
            if (is_suppressed[i]) continue;
            final_faces.push_back(proposals[i]);
            float area_i = (proposals[i].bbox[2] - proposals[i].bbox[0]) * (proposals[i].bbox[3] - proposals[i].bbox[1]);
            for (size_t j = i + 1; j < proposals.size(); ++j) {
                if (is_suppressed[j]) continue;
                float ix1 = std::max(proposals[i].bbox[0], proposals[j].bbox[0]);
                float iy1 = std::max(proposals[i].bbox[1], proposals[j].bbox[1]);
                float ix2 = std::min(proposals[i].bbox[2], proposals[j].bbox[2]);
                float iy2 = std::min(proposals[i].bbox[3], proposals[j].bbox[3]);
                float inter_w = std::max(0.0f, ix2 - ix1);
                float inter_h = std::max(0.0f, iy2 - iy1);
                float inter_area = inter_w * inter_h;
                float area_j = (proposals[j].bbox[2] - proposals[j].bbox[0]) * (proposals[j].bbox[3] - proposals[j].bbox[1]);
                float union_area = area_i + area_j - inter_area;
                float iou = (union_area > 1e-5f) ? inter_area / union_area : 0.0f;
                if (iou > iou_threshold) {
                    is_suppressed[j] = true;
                }
            }
        }
    }
}

struct FaceDetector::Impl {
    std::unique_ptr<Ort::Session> det_session;
    Ort::MemoryInfo memory_info{nullptr};
    std::vector<uint8_t> frame_cpu_buffer_rgba;
    std::vector<uint8_t> bgr_buffer;
    std::vector<uint8_t> resized_buffer;
    std::vector<uint8_t> letterbox_buffer;
    std::vector<float> blob;
    std::vector<FaceResult> proposals;
    size_t m_lastFaceCount = -1; 

    Impl() : memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}
};

FaceDetector::FaceDetector() : pImpl(std::make_unique<Impl>()) {}
FaceDetector::~FaceDetector() = default;

// MODIFIED: Signature updated to match base class. The console parameter is currently unused but is required.
HRESULT FaceDetector::Initialize(OnnxRuntime& onnx, const std::string& model_dir, Console* console) {
    try {
        std::wstring model_path = WinTegrity::ToWide(model_dir + "/det_10g.onnx");
        pImpl->det_session = onnx.CreateSession(model_path);
    } catch (const std::exception& e) {
        if (console) console->AddLog("[ERROR] FaceDetector failed to load model: %s", e.what());
        OutputDebugStringA(e.what());
        return E_FAIL;
    }
    return S_OK;
}

void FaceDetector::Process(FrameData& frameData) {
    frameData.faces.clear();
    if (!pImpl->det_session || !frameData.pInputSRV) {
        if (frameData.pConsole && pImpl->m_lastFaceCount != 0) {
             frameData.pConsole->AddLog("No faces detected.");
             pImpl->m_lastFaceCount = 0;
        }
        return;
    }

    WinTegrity::TegrityImageBuffer cpu_frame_rgba;
    if (FAILED(GetCpuImageFromGpuTexture(frameData.pDevice, frameData.pContext, frameData.pInputSRV, pImpl->frame_cpu_buffer_rgba, cpu_frame_rgba))) {
        return;
    }
    
    pImpl->bgr_buffer.resize(static_cast<size_t>(cpu_frame_rgba.width) * cpu_frame_rgba.height * 3);
    WinTegrity::TegrityImageBuffer bgr_img = { pImpl->bgr_buffer.data(), cpu_frame_rgba.width, cpu_frame_rgba.height, 3 };
    #pragma omp parallel for
    for (int i = 0; i < bgr_img.width * bgr_img.height; ++i) {
        memcpy(bgr_img.data + (size_t)i * 3, cpu_frame_rgba.data + (size_t)i * 4, 3);
    }
    
    const int input_size = 640;
    float r = std::min(static_cast<float>(input_size) / bgr_img.width, static_cast<float>(input_size) / bgr_img.height);
    int in_w = static_cast<int>(bgr_img.width * r);
    int in_h = static_cast<int>(bgr_img.height * r);

    pImpl->resized_buffer.resize(static_cast<size_t>(in_w) * in_h * 3);
    WinTegrity::TegrityImageBuffer resized_img = { pImpl->resized_buffer.data(), in_w, in_h, 3 };
    Files::ResizeImage(bgr_img, resized_img);

    pImpl->letterbox_buffer.assign(static_cast<size_t>(input_size) * input_size * 3, 114); // Pad with gray
    WinTegrity::TegrityImageBuffer letterbox_img = { pImpl->letterbox_buffer.data(), input_size, input_size, 3 };
    for (int y = 0; y < in_h; ++y) {
        unsigned char* dst_ptr = letterbox_img.data + (static_cast<size_t>(y) * letterbox_img.width * 3);
        unsigned char* src_ptr = resized_img.data + (static_cast<size_t>(y) * resized_img.width * 3);
        memcpy(dst_ptr, src_ptr, static_cast<size_t>(resized_img.width) * 3);
    }

    CreateBlobFromBGR(letterbox_img, pImpl->blob, 1.0f / 128.0f, 127.5f, 127.5f, 127.5f);

    const char* input_names[] = {"input.1"};
    const char* output_names[] = {"448", "471", "494", "451", "474", "497", "454", "477", "500"};
    std::vector<int64_t> input_shape = {1, 3, input_size, input_size};
    
    try {
        auto input_tensor = Ort::Value::CreateTensor<float>(pImpl->memory_info, pImpl->blob.data(), pImpl->blob.size(), input_shape.data(), input_shape.size());
        auto output_tensors = pImpl->det_session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 9);
        pImpl->proposals.clear();
        const int strides[] = {8, 16, 32};
        for (int i = 0; i < 3; ++i) {
            int stride = strides[i];
            const float* scores = output_tensors[i].GetTensorData<float>();
            const float* bbox_preds = output_tensors[i + 3].GetTensorData<float>();
            const float* kps_preds = output_tensors[i + 6].GetTensorData<float>();
            int height = input_size / stride;
            int width = input_size / stride;
            int num_anchors = height * width * 2;
            for (int anchor_idx = 0; anchor_idx < num_anchors; ++anchor_idx) {
                if (scores[anchor_idx] > 0.5f) {
                    FaceResult face = {};
                    face.detection_score = scores[anchor_idx];
                    int grid_idx = anchor_idx / 2;
                    float acy = static_cast<float>(grid_idx / width) * stride;
                    float acx = static_cast<float>(grid_idx % width) * stride;

                    face.bbox[0] = (acx - bbox_preds[anchor_idx * 4 + 0] * stride) / r;
                    face.bbox[1] = (acy - bbox_preds[anchor_idx * 4 + 1] * stride) / r;
                    face.bbox[2] = (acx + bbox_preds[anchor_idx * 4 + 2] * stride) / r;
                    face.bbox[3] = (acy + bbox_preds[anchor_idx * 4 + 3] * stride) / r;
                    
                    for (int k = 0; k < 5; ++k) {
                        float kps_x = (acx + kps_preds[anchor_idx * 10 + k * 2] * stride);
                        float kps_y = (acy + kps_preds[anchor_idx * 10 + k * 2 + 1] * stride);
                        face.landmarks(k, 0) = kps_x / r;
                        face.landmarks(k, 1) = kps_y / r;
                    }
                    pImpl->proposals.push_back(face);
                }
            }
        }
        NMS(pImpl->proposals, 0.4f, frameData.faces);

        if (frameData.pConsole && frameData.faces.size() != pImpl->m_lastFaceCount) {
            if (frameData.faces.empty()) {
                frameData.pConsole->AddLog("No faces detected.");
            } else {
                frameData.pConsole->AddLog("Face detection found %zu face(s).", frameData.faces.size());
            }
        }
        pImpl->m_lastFaceCount = frameData.faces.size();

    } catch (const Ort::Exception& e) {
        if (frameData.pConsole) {
            frameData.pConsole->AddLog("ONNX Error: %s", e.what());
        }
    }
}