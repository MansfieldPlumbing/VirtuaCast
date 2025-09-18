// File: src/VirtuaCast/DebugOutput.cpp
#include "DebugOutput.h"
#include "VirtuaCast.h" 
#include "Algorithms.h"
#include "Files.h"
#include "SourceFace.h"
#include "FaceDetection.h"
#include "FaceEmbedding.h"
#include "FaceSwap.h"
#include "OnnxRuntime.h"
#include "DMLHelper.h"
#include "Console.h"

#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <d3d11.h>
#include <d3d12.h>
#include "directx/d3dx12.h"
#include <wrl/client.h>
#include <algorithm>
#include <fstream> 
#include <iomanip> 

using namespace Microsoft::WRL;
using namespace VirtuaCast;

namespace {

HRESULT SavePlanarTextureAsImage_Debug(
    ID3D11Device* pDevice,
    ID3D11DeviceContext* pContext,
    ID3D11Texture2D* pPlanarTexture,
    UINT width, UINT height,
    const std::wstring& filepath)
{
    if (!pDevice || !pContext || !pPlanarTexture) return E_INVALIDARG;

    D3D11_TEXTURE2D_DESC desc;
    pPlanarTexture->GetDesc(&desc);

    D3D11_TEXTURE2D_DESC stagingDesc = desc;
    stagingDesc.Usage = D3D11_USAGE_STAGING;
    stagingDesc.BindFlags = 0;
    stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    stagingDesc.MiscFlags = 0;

    ComPtr<ID3D11Texture2D> stagingTexture;
    HRESULT hr = pDevice->CreateTexture2D(&stagingDesc, nullptr, &stagingTexture);
    if (FAILED(hr)) return hr;

    pContext->CopyResource(stagingTexture.Get(), pPlanarTexture);

    D3D11_MAPPED_SUBRESOURCE mappedResource;
    hr = pContext->Map(stagingTexture.Get(), 0, D3D11_MAP_READ, 0, &mappedResource);
    if (FAILED(hr)) return hr;

    std::vector<unsigned char> bgr_buffer(static_cast<size_t>(width) * height * 3);
    const float* pSrcFloat = static_cast<const float*>(mappedResource.pData);
    const size_t rowPitchInFloats = mappedResource.RowPitch / sizeof(float);

    #pragma omp parallel for
    for (int y = 0; y < static_cast<int>(height); ++y) {
        for (int x = 0; x < static_cast<int>(width); ++x) {
            float r = pSrcFloat[(size_t)y * rowPitchInFloats + x];
            float g = pSrcFloat[((size_t)y + height) * rowPitchInFloats + x];
            float b = pSrcFloat[((size_t)y + 2 * height) * rowPitchInFloats + x];

            size_t dst_idx = (static_cast<size_t>(y) * width + x) * 3;
            bgr_buffer[dst_idx + 0] = static_cast<unsigned char>(std::clamp(b * 255.0f, 0.0f, 255.0f));
            bgr_buffer[dst_idx + 1] = static_cast<unsigned char>(std::clamp(g * 255.0f, 0.0f, 255.0f));
            bgr_buffer[dst_idx + 2] = static_cast<unsigned char>(std::clamp(r * 255.0f, 0.0f, 255.0f));
        }
    }

    pContext->Unmap(stagingTexture.Get(), 0);

    WinTegrity::TegrityImageBuffer image_view = { bgr_buffer.data(), (int)width, (int)height, 3 };
    return Files::WriteImage(filepath, image_view);
}

}

static HRESULT SaveEmbeddingToText(const std::vector<float>& embedding, const std::wstring& filepath) {
    if (embedding.size() != 512) {
        return E_INVALIDARG;
    }
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return E_FAIL;
    }
    file << std::fixed << std::setprecision(8);
    for (size_t i = 0; i < embedding.size(); ++i) {
        file << embedding[i] << std::endl;
    }
    file.close();
    return S_OK;
}

static HRESULT SaveFloatBufferToText(const float* float_data, size_t element_count, const std::wstring& filepath) {
    if (!float_data) return E_POINTER;
    std::ofstream file(filepath);
    if (!file.is_open()) return E_FAIL;
    file << std::fixed << std::setprecision(8);
    for (size_t i = 0; i < element_count; ++i) {
        file << float_data[i] << std::endl;
    }
    file.close();
    return S_OK;
}

static HRESULT GetCpuImageFromGpuTexture(
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
    
    UINT bytesPerPixel = 0;
    switch (desc.Format) {
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM:
            bytesPerPixel = 4;
            break;
        case DXGI_FORMAT_R8_UNORM:
            bytesPerPixel = 1;
            break;
        default: return E_FAIL;
    }

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

static HRESULT GetCpuImageFromGpuPlanarTexture(
    ID3D11Device* pDevice,
    ID3D11DeviceContext* pContext,
    ID3D11ShaderResourceView* pSrv,
    std::vector<unsigned char>& out_cpu_data,
    WinTegrity::TegrityImageBuffer& out_buffer_view,
    UINT width, UINT height)
{
    if (!pDevice || !pContext || !pSrv) return E_INVALIDARG;

    ComPtr<ID3D11Resource> resource;
    pSrv->GetResource(&resource);

    ComPtr<ID3D11Texture2D> texture;
    HRESULT hr = resource.As(&texture);
    if (FAILED(hr)) return E_NOINTERFACE;

    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);

    if (desc.Format != DXGI_FORMAT_R32_FLOAT || desc.Width != width || desc.Height != height * 3) {
        return E_INVALIDARG;
    }

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

    const size_t num_pixels = static_cast<size_t>(width) * height;
    out_cpu_data.resize(num_pixels * 3);
    
    out_buffer_view.data = out_cpu_data.data();
    out_buffer_view.width = width;
    out_buffer_view.height = height;
    out_buffer_view.channels = 3;

    const float* pSrcFloat = static_cast<const float*>(mappedResource.pData);
    const size_t rowPitchInFloats = mappedResource.RowPitch / sizeof(float);

    #pragma omp parallel for
    for (int y = 0; y < (int)height; ++y) {
        for (int x = 0; x < (int)width; ++x) {
            float r_val = pSrcFloat[ (size_t)y * rowPitchInFloats + x ];
            float g_val = pSrcFloat[ ((size_t)y + height) * rowPitchInFloats + x ];
            float b_val = pSrcFloat[ ((size_t)y + 2 * height) * rowPitchInFloats + x ];
            size_t dst_idx = ((size_t)y * width + x) * 3;
            out_buffer_view.data[dst_idx + 0] = static_cast<unsigned char>(std::clamp(b_val * 255.0f, 0.0f, 255.0f));
            out_buffer_view.data[dst_idx + 1] = static_cast<unsigned char>(std::clamp(g_val * 255.0f, 0.0f, 255.0f));
            out_buffer_view.data[dst_idx + 2] = static_cast<unsigned char>(std::clamp(r_val * 255.0f, 0.0f, 255.0f));
        }
    }

    pContext->Unmap(stagingTexture.Get(), 0);
    return S_OK;
}

HRESULT TegrityDebug_RunFullSwapTrace(void* app_handle, const TegritySourceFace* source_face) {
    if (!app_handle || !source_face) return E_POINTER;
    
    auto* app = static_cast<VirtuaCast::Application*>(app_handle);
    auto* detector = app->GetFaceDetector();
    auto* embedder = app->GetFaceEmbedder();
    auto* swapper = app->GetFaceSwap();
    auto* onnx = app->GetOnnxRuntime();
    ID3D11Device* device = app->GetDevice();
    ID3D11DeviceContext* context = app->GetContext();
    ID3D11ShaderResourceView* current_frame_srv = app->GetFinalFrameSRV();
    Console* console = app->GetConsole();

    if (!detector || !embedder || !swapper || !device || !context || !current_frame_srv || !console || !onnx) return E_FAIL;

    console->AddLog("\n--- [VirtuaCast Debug] STARTING ATOMIC SWAP TRACE ---");
    
    console->AddLog("  [C++ DEBUG] Received trace request for '%s'", source_face->name);
    
    std::vector<unsigned char> dump_buffer_rgba;
    WinTegrity::TegrityImageBuffer dump_view_rgba;
    if (SUCCEEDED(GetCpuImageFromGpuTexture(device, context, current_frame_srv, dump_buffer_rgba, dump_view_rgba)))
    {
        std::vector<unsigned char> dump_buffer_bgr(static_cast<size_t>(dump_view_rgba.width) * dump_view_rgba.height * 3);
        WinTegrity::TegrityImageBuffer dump_view_bgr = { dump_buffer_bgr.data(), dump_view_rgba.width, dump_view_rgba.height, 3 };
        #pragma omp parallel for
        for (int i = 0; i < dump_view_bgr.width * dump_view_bgr.height; ++i) {
            memcpy(dump_view_bgr.data + (size_t)i * 3, dump_view_rgba.data + (size_t)i * 4, 3);
        }
        Files::WriteImage(L"DEBUG_01b_PipelineInputFrame.jpg", dump_view_bgr);
        console->AddLog("  [Debug] Saved DEBUG_01b_PipelineInputFrame.jpg to inspect pipeline input.");
    }
    else
    {
        console->AddLog("  [Debug][ERROR] Failed to save intermediate pipeline input frame.");
    }

    WinTegrity::TegrityImageBuffer src_img_orig_bgr;
    std::string name_str(source_face->name);
    std::wstring base_path = L"sources\\" + WinTegrity::ToWide(name_str);
    
    if (FAILED(Files::ReadImageToBGR((base_path + L".jpg").c_str(), src_img_orig_bgr)) &&
        FAILED(Files::ReadImageToBGR((base_path + L".png").c_str(), src_img_orig_bgr)) &&
        FAILED(Files::ReadImageToBGR((base_path + L".jpeg").c_str(), src_img_orig_bgr))) {
        console->AddLog("  [Debug] FATAL: Could not find source image file for '%s'. Aborting trace.", name_str.c_str());
        return E_INVALIDARG;
    }

    {
        std::vector<unsigned char> resize_buffer(256 * 256 * src_img_orig_bgr.channels);
        WinTegrity::TegrityImageBuffer resized_src_img = { resize_buffer.data(), 256, 256, src_img_orig_bgr.channels };
        Files::ResizeImage(src_img_orig_bgr, resized_src_img);
        Files::WriteImage(L"DEBUG_00_Source_Face.jpg", resized_src_img);
        console->AddLog("  [1/15] Saved DEBUG_00_Source_Face.jpg");
    }
    
    ComPtr<ID3D11Texture2D> temp_src_tex;
    ComPtr<ID3D11ShaderResourceView> temp_src_srv;
    {
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = src_img_orig_bgr.width;
        desc.Height = src_img_orig_bgr.height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

        std::vector<uint8_t> bgra_buffer( (size_t)desc.Width * desc.Height * 4);
        #pragma omp parallel for
        for(int i = 0; i < (int)desc.Width * (int)desc.Height; ++i) {
            bgra_buffer[(size_t)i*4+0] = src_img_orig_bgr.data[(size_t)i*3+0];
            bgra_buffer[(size_t)i*4+1] = src_img_orig_bgr.data[(size_t)i*3+1];
            bgra_buffer[(size_t)i*4+2] = src_img_orig_bgr.data[(size_t)i*3+2];
            bgra_buffer[(size_t)i*4+3] = 255;
        }
        
        D3D11_SUBRESOURCE_DATA subData = { bgra_buffer.data(), (UINT)desc.Width * 4, 0 };
        device->CreateTexture2D(&desc, &subData, &temp_src_tex);
        device->CreateShaderResourceView(temp_src_tex.Get(), nullptr, &temp_src_srv);
    }

    FrameData src_frame_data;
    src_frame_data.pDevice = device;
    src_frame_data.pContext = context;
    src_frame_data.pInputSRV = temp_src_srv.Get();
    src_frame_data.pConsole = console;
    detector->Process(src_frame_data);

    if (!src_frame_data.faces.empty()) {
        const auto& src_face_details = src_frame_data.faces.front();
        Files::DrawRectangle(src_img_orig_bgr, (int)src_face_details.bbox[0], (int)src_face_details.bbox[1], (int)src_face_details.bbox[2], (int)src_face_details.bbox[3], 255, 0, 0, 2);
        for (int i = 0; i < 5; ++i) {
            Files::DrawFilledRectangle(src_img_orig_bgr, (int)src_face_details.landmarks(i, 0) - 2, (int)src_face_details.landmarks(i, 1) - 2, (int)src_face_details.landmarks(i, 0) + 2, (int)src_face_details.landmarks(i, 1) + 2, 0, 255, 0);
        }
        Files::WriteImage(L"DEBUG_00a_Source_Face_With_Detection.jpg", src_img_orig_bgr);
        console->AddLog("  [2/15] Saved DEBUG_00a_Source_Face_With_Detection.jpg");

        const int align_size = 112;
        std::vector<uint8_t> aligned_buffer(align_size * align_size * 3);
        WinTegrity::TegrityImageBuffer aligned_img_dst = {aligned_buffer.data(), align_size, align_size, 3};
        Eigen::Matrix<double, 5, 2> arcface_dst_pts_112;
        arcface_dst_pts_112 << 38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366, 41.5493, 92.3655, 70.7299, 92.2041;
        Eigen::Matrix<double, 2, 3> M = WinTegrity::estimateSimilarityTransform(src_face_details.landmarks, arcface_dst_pts_112);
        WinTegrity::warpAffineBilinear(src_img_orig_bgr, aligned_img_dst, M, nullptr);
        Files::WriteImage(L"DEBUG_01_Aligned_Source.jpg", aligned_img_dst);
        console->AddLog("  [3/15] Saved DEBUG_01_Aligned_Source.jpg");
        
        std::vector<float> temp_embedding(source_face->normed_embedding, source_face->normed_embedding + 512);
        if (SUCCEEDED(SaveEmbeddingToText(temp_embedding, L"DEBUG_01a_Source_Embedding.txt"))) {
            console->AddLog("  [4/15] Saved DEBUG_01a_Source_Embedding.txt");
        } else {
            console->AddLog("  [ERROR] Failed to save source embedding text file.");
        }
    } else {
        console->AddLog("  [WARN] Could not find face in source image to generate detection/alignment. Skipping steps.");
    }
    Files::FreeImageBuffer(src_img_orig_bgr);

    std::vector<unsigned char> frame_rgba_cpu_buffer;
    WinTegrity::TegrityImageBuffer frame_rgba_view;
    if(FAILED(GetCpuImageFromGpuTexture(device, context, current_frame_srv, frame_rgba_cpu_buffer, frame_rgba_view))) {
        return E_FAIL;
    }
    
    std::vector<unsigned char> bgr_buffer(static_cast<size_t>(frame_rgba_view.width)* frame_rgba_view.height * 3);
    WinTegrity::TegrityImageBuffer frame_bgr = { bgr_buffer.data(), frame_rgba_view.width, frame_rgba_view.height, 3 };
    #pragma omp parallel for
    for (int i = 0; i < frame_bgr.width * frame_bgr.height; ++i) {
        memcpy(frame_bgr.data + (size_t)i * 3, frame_rgba_view.data + (size_t)i * 4, 3);
    }
    
    FrameData target_frame_data;
    target_frame_data.pDevice = device;
    target_frame_data.pContext = context;
    target_frame_data.pInputSRV = current_frame_srv;
    target_frame_data.pConsole = console;
    detector->Process(target_frame_data);
    if (target_frame_data.faces.empty()) {
        console->AddLog("  [Debug] No target face found in frame. Aborting trace.");
        return S_OK; 
    }
    const auto& target_face = target_frame_data.faces.front();
    
    Files::WriteImage(L"DEBUG_02_Frame_With_Target.jpg", frame_bgr);
    console->AddLog("  [5/15] Saved DEBUG_02_Frame_With_Target.jpg");

    const int align_size_target = 128;
    std::vector<uint8_t> aligned_target_buffer(static_cast<size_t>(align_size_target) * align_size_target * 3);
    WinTegrity::TegrityImageBuffer aligned_target_img = {aligned_target_buffer.data(), align_size_target, align_size_target, 3};
    Eigen::Matrix<double, 5, 2> arcface_dst_pts;
    arcface_dst_pts << 38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366, 41.5493, 92.3655, 70.7299, 92.2041;
    Eigen::Matrix<double, 2, 3> M_align = WinTegrity::estimateSimilarityTransform(target_face.landmarks, arcface_dst_pts);
    WinTegrity::warpAffineBilinear(frame_bgr, aligned_target_img, M_align, nullptr);
    Files::WriteImage(L"DEBUG_03_Aligned_Target_Pre_ONNX.jpg", aligned_target_img);
    console->AddLog("  [6/15] Saved DEBUG_03_Aligned_Target_Pre_ONNX.jpg");

    embedder->Process(target_frame_data);
    swapper->Process(target_frame_data);
    console->AddLog("  [7/15] GPU pipeline executed up to ONNX Run call.");

    if (SUCCEEDED(SavePlanarTextureAsImage_Debug(device, context, swapper->GetModelInputTexture(), 128, 128, L"DEBUG_02a_Warped_Face_Pre_ONNX.jpg"))) {
        console->AddLog("  [Debug] Saved DEBUG_02a_Warped_Face_Pre_ONNX.jpg to inspect WarpAffine output.");
    } else {
        console->AddLog("  [Debug][ERROR] Failed to save intermediate warped face texture.");
    }
    
    ID3D12Fence* completionFence = swapper->GetCompletionFence();
    UINT64 fenceValue = swapper->GetLastFenceValue();
    if (completionFence && fenceValue > 0)
    {
        HANDLE fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (fenceEvent)
        {
            if (SUCCEEDED(completionFence->SetEventOnCompletion(fenceValue, fenceEvent)))
            {
                WaitForSingleObject(fenceEvent, INFINITE);
            }
            CloseHandle(fenceEvent);
        }
    }

    ID3D12Resource* input_buffer = swapper->GetPrivateInputBuffer();
    if (input_buffer) {
        ID3D12Device* d3d12Device = onnx->GetDMLHelper().GetD3D12Device();
        ID3D12CommandQueue* commandQueue = onnx->GetDMLHelper().GetCommandQueue();
        ComPtr<ID3D12CommandAllocator> cmdAllocator;
        ComPtr<ID3D12GraphicsCommandList> cmdList;
        d3d12Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&cmdAllocator));
        d3d12Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, cmdAllocator.Get(), nullptr, IID_PPV_ARGS(&cmdList));

        const UINT64 bufferSize = input_buffer->GetDesc().Width;
        ComPtr<ID3D12Resource> readbackBuffer;
        auto readbackHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
        auto readbackDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);
        d3d12Device->CreateCommittedResource(&readbackHeap, D3D12_HEAP_FLAG_NONE, &readbackDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readbackBuffer));

        cmdList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(input_buffer, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE));
        cmdList->CopyResource(readbackBuffer.Get(), input_buffer);
        cmdList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(input_buffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON));
        cmdList->Close();
        ID3D12CommandList* lists_to_execute[] = { cmdList.Get() };
        commandQueue->ExecuteCommandLists(1, lists_to_execute);
        
        onnx->GetDMLHelper().WaitForGpuIdle();

        void* pData;
        readbackBuffer->Map(0, nullptr, &pData);
        if (SUCCEEDED(SaveFloatBufferToText(static_cast<float*>(pData), bufferSize / sizeof(float), L"DEBUG_03a_ONNX_Input_Raw.txt"))) {
            console->AddLog("  [8/15] Saved DEBUG_03a_ONNX_Input_Raw.txt");
        }
        readbackBuffer->Unmap(0, nullptr);
    } else {
        console->AddLog("  [ERROR] Could not retrieve private input buffer for dumping.");
    }
    
    ID3D12Resource* output_buffer = swapper->GetPrivateOutputBuffer();
    if (output_buffer) {
        ID3D12Device* d3d12Device = onnx->GetDMLHelper().GetD3D12Device();
        ID3D12CommandQueue* commandQueue = onnx->GetDMLHelper().GetCommandQueue();
        ComPtr<ID3D12CommandAllocator> cmdAllocator;
        ComPtr<ID3D12GraphicsCommandList> cmdList;
        d3d12Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&cmdAllocator));
        d3d12Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, cmdAllocator.Get(), nullptr, IID_PPV_ARGS(&cmdList));

        const UINT64 bufferSize = output_buffer->GetDesc().Width;
        ComPtr<ID3D12Resource> readbackBuffer;
        auto readbackHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
        auto readbackDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);
        d3d12Device->CreateCommittedResource(&readbackHeap, D3D12_HEAP_FLAG_NONE, &readbackDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readbackBuffer));

        cmdList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(output_buffer, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE));
        cmdList->CopyResource(readbackBuffer.Get(), output_buffer);
        cmdList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(output_buffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON));
        cmdList->Close();
        ID3D12CommandList* lists_to_execute[] = { cmdList.Get() };
        commandQueue->ExecuteCommandLists(1, lists_to_execute);
        
        onnx->GetDMLHelper().WaitForGpuIdle();

        void* pData;
        readbackBuffer->Map(0, nullptr, &pData);
        if (SUCCEEDED(SaveFloatBufferToText(static_cast<float*>(pData), bufferSize / sizeof(float), L"DEBUG_04a_ONNX_Output_Raw.txt"))) {
            console->AddLog("  [9/15] Saved DEBUG_04a_ONNX_Output_Raw.txt");
        }
        readbackBuffer->Unmap(0, nullptr);
    } else {
        console->AddLog("  [ERROR] Could not retrieve private output buffer for dumping.");
    }
    
    console->AddLog("  [10/15] ONNX model executed on GPU.");
    
    ID3D11ShaderResourceView* swapped_face_srv = swapper->GetModelOutputSRV();
    if (swapped_face_srv) {
        std::vector<unsigned char> swapped_face_cpu_buffer;
        WinTegrity::TegrityImageBuffer swapped_face_view;
        if (SUCCEEDED(GetCpuImageFromGpuPlanarTexture(device, context, swapped_face_srv, swapped_face_cpu_buffer, swapped_face_view, 128, 128))) {
            Files::WriteImage(L"DEBUG_04_Generated_Swap_Face.jpg", swapped_face_view);
            console->AddLog("  [11/15] Saved DEBUG_04_Generated_Swap_Face.jpg");
        } else {
             console->AddLog("  [ERROR] Failed to get CPU image from GPU planar texture for swapped face.");
        }
    }
    
    ID3D11ShaderResourceView* mask_srv = swapper->GetMaskSRV();
    if (mask_srv) {
        std::vector<unsigned char> mask_cpu_buffer;
        WinTegrity::TegrityImageBuffer mask_view;
        if (SUCCEEDED(GetCpuImageFromGpuTexture(device, context, mask_srv, mask_cpu_buffer, mask_view))) {
             Files::WriteImage(L"DEBUG_05_Blending_Mask.jpg", mask_view);
             console->AddLog("  [12/15] Saved DEBUG_05_Blending_Mask.jpg");
        } else {
             console->AddLog("  [ERROR] Failed to get CPU image from GPU texture for blending mask.");
        }
    }

    Eigen::Matrix<double, 2, 3> M_paste = WinTegrity::invertAffineTransform(M_align);
    std::vector<uint8_t> warped_mask_buffer(frame_bgr.width * frame_bgr.height, 0);
    WinTegrity::TegrityImageBuffer warped_mask = { warped_mask_buffer.data(), frame_bgr.width, frame_bgr.height, 1 };
    
    WinTegrity::TegrityDetectedFace legacy_target_face;
    memcpy(legacy_target_face.bbox, target_face.bbox, sizeof(float) * 4);
    
    WinTegrity::TegrityRect target_roi = WinTegrity::get_face_roi(legacy_target_face, frame_bgr.width, frame_bgr.height);
    
    std::vector<uint8_t> square_mask_buffer(128 * 128, 255);
    WinTegrity::TegrityImageBuffer square_mask = { square_mask_buffer.data(), 128, 128, 1 };
    WinTegrity::warpAffineBilinear(square_mask, warped_mask, M_paste, &target_roi);
    Files::WriteImage(L"DEBUG_06_Warped_Mask_To_Canvas.jpg", warped_mask);
    console->AddLog("  [13/15] Saved DEBUG_06_Warped_Mask_To_Canvas.jpg");
    
    WinTegrity::TegrityPipelineConfig dummy_config;
    auto params = WinTegrity::tuneBlendingParameters((int)(target_face.bbox[2]-target_face.bbox[0]), dummy_config);
    WinTegrity::applyGaussianMaskBlur(warped_mask, params.blur_kernel_radius, params.blur_sigma);
    Files::WriteImage(L"DEBUG_07_Final_Processed_Mask.jpg", warped_mask);
    console->AddLog("  [14/15] Saved DEBUG_07_Final_Processed_Mask.jpg");
    
    ID3D11ShaderResourceView* final_swapped_srv = swapper->GetGeneratedSwappedFaceSRV();
    if (!final_swapped_srv) {
        final_swapped_srv = app->GetFinalFrameSRV();
    }
    
    std::vector<unsigned char> final_frame_rgba_cpu_buffer;
    WinTegrity::TegrityImageBuffer final_frame_rgba_view;
    GetCpuImageFromGpuTexture(device, context, final_swapped_srv, final_frame_rgba_cpu_buffer, final_frame_rgba_view);
    
    std::vector<unsigned char> final_bgr_buffer(static_cast<size_t>(final_frame_rgba_view.width)* final_frame_rgba_view.height * 3);
    WinTegrity::TegrityImageBuffer final_frame_bgr = { final_bgr_buffer.data(), final_frame_rgba_view.width, final_frame_rgba_view.height, 3 };
    #pragma omp parallel for
    for (int i = 0; i < final_frame_bgr.width * final_frame_bgr.height; ++i) {
        memcpy(final_frame_bgr.data + (size_t)i * 3, final_frame_rgba_view.data + (size_t)i * 4, 3);
    }
    Files::WriteImage(L"DEBUG_08_Final_Result.jpg", final_frame_bgr);
    console->AddLog("  [15/15] Saved DEBUG_08_Final_Result.jpg");

    console->AddLog("--- [VirtuaCast Debug] TRACE COMPLETE ---\n");
    return S_OK;
}