#include "FaceSwap.h"
#include "OnnxRuntime.h"
#include "DMLHelper.h"
#include "Algorithms.h"
#include "Types.h"
#include "WarpAffine.h"
#include "Blend.h"
#include "Console.h"
#include "GpuTypes.h"
#include "Files.h"
#include <d3d11_4.h>
#include <d3d12.h>
#include "directx/d3dx12.h"
#include <wrl/client.h>
#include <Eigen/Dense>
#include <vector>
#include <fstream> 

using namespace Microsoft::WRL;
using namespace VirtuaCast;

namespace VirtuaCast {

struct FaceSwap::Impl {
    std::unique_ptr<Ort::Session> swap_session;
    OnnxRuntime* m_onnx = nullptr;
    bool hasSourceFace = false;
    std::vector<float> sourceEmbedding;
    
    Eigen::MatrixXf emap;

    ComPtr<ID3D11Device> m_device;
    ComPtr<ID3D11Device5> m_device5;
    ComPtr<ID3D11DeviceContext> m_context;
    ComPtr<ID3D11DeviceContext4> m_context4;
    ComPtr<ID3D12CommandAllocator> m_d3d12CmdAllocator;
    ComPtr<ID3D12GraphicsCommandList> m_d3d12CmdList;


    std::unique_ptr<WarpAffine> m_warp_affine;
    std::unique_ptr<Blend> m_blend;

    ComPtr<ID3D11Texture2D> m_modelInputTexture;
    ComPtr<ID3D11UnorderedAccessView> m_modelInputUAV;
    ComPtr<ID3D11Texture2D> m_modelOutputTexture;
    ComPtr<ID3D11ShaderResourceView> m_modelOutputSRV;
    ComPtr<ID3D12Resource> m_modelInputD3D12;
    ComPtr<ID3D12Resource> m_modelOutputD3D12;

    ComPtr<ID3D12Resource> m_privateModelInputBuffer;
    ComPtr<ID3D12Resource> m_privateModelOutputBuffer;

    ComPtr<ID3D11Texture2D> m_mask_tex;
    ComPtr<ID3D11ShaderResourceView> m_mask_srv;
    
    ComPtr<ID3D11Texture2D> m_generatedSwappedFaceTexture;
    ComPtr<ID3D11RenderTargetView> m_generatedSwappedFaceRTV;
    ComPtr<ID3D11ShaderResourceView> m_generatedSwappedFaceSRV;
    ComPtr<ID3D11UnorderedAccessView> m_generatedSwappedFaceUAV;

    ComPtr<ID3D11Fence> m_d3d11Fence;
    ComPtr<ID3D12Fence> m_d3d12Fence;
    UINT64 m_fenceValue = 0;
};

FaceSwap::FaceSwap() : pImpl(std::make_unique<Impl>()) {}
FaceSwap::~FaceSwap() = default;

HRESULT FaceSwap::Initialize(OnnxRuntime& onnx, const std::string& model_dir, Console* console) {
    pImpl->m_onnx = &onnx;
    std::wstring swap_model_path = WinTegrity::ToWide(model_dir + "/inswapper_128.onnx");
    pImpl->swap_session = onnx.CreateSession(swap_model_path);

    pImpl->emap.resize(512, 512);

    std::string emap_path = model_dir + "/emap.bin";
    std::ifstream emap_file(emap_path, std::ios::binary);
    if (!emap_file.is_open()) {
        if (console) console->AddLog("[FATAL] emap.bin not found. Please run scripts/extract_emap.py");
        return E_FAIL;
    }

    const size_t emap_byte_size = 512 * 512 * sizeof(float);
    emap_file.read(reinterpret_cast<char*>(pImpl->emap.data()), emap_byte_size);
    if (emap_file.gcount() != emap_byte_size) {
        if (console) console->AddLog("[FATAL] emap.bin is corrupt or has the wrong size.");
        emap_file.close();
        return E_FAIL;
    }
    emap_file.close();
    
    return S_OK;
}

void FaceSwap::SetSourceFace(const std::vector<float>& embedding) {
    if (embedding.size() != 512) {
        pImpl->hasSourceFace = false;
        return;
    }
    pImpl->sourceEmbedding = embedding;
    pImpl->hasSourceFace = true;
}

void FaceSwap::ClearSourceFace() {
    pImpl->hasSourceFace = false;
}

ID3D11ShaderResourceView* FaceSwap::GetGeneratedSwappedFaceSRV() const
{
    return pImpl->m_generatedSwappedFaceSRV.Get();
}

ID3D11Texture2D* FaceSwap::GetGeneratedSwappedFaceTexture() const
{
    return pImpl->m_generatedSwappedFaceTexture.Get();
}


ID3D11ShaderResourceView* FaceSwap::GetModelOutputSRV() const
{
    return pImpl->m_modelOutputSRV.Get();
}

ID3D11ShaderResourceView* FaceSwap::GetMaskSRV() const
{
    return pImpl->m_mask_srv.Get();
}

void FaceSwap::Process(FrameData& frameData)
{
    if (!pImpl->m_warp_affine)
    {
        pImpl->m_device = frameData.pDevice;
        pImpl->m_context = frameData.pContext;
        pImpl->m_device.As(&pImpl->m_device5);
        pImpl->m_context.As(&pImpl->m_context4);
        pImpl->m_onnx->GetDMLHelper().GetD3D12Device()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&pImpl->m_d3d12CmdAllocator));
        pImpl->m_onnx->GetDMLHelper().GetD3D12Device()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, pImpl->m_d3d12CmdAllocator.Get(), nullptr, IID_PPV_ARGS(&pImpl->m_d3d12CmdList));
        pImpl->m_d3d12CmdList->Close();

        pImpl->m_warp_affine = std::make_unique<WarpAffine>();
        pImpl->m_warp_affine->Initialize(pImpl->m_device.Get());
        pImpl->m_blend = std::make_unique<Blend>();
        pImpl->m_blend->Initialize(pImpl->m_device.Get());

        D3D11_TEXTURE2D_DESC texDesc = {};
        texDesc.Width = 128;
        texDesc.Height = 128 * 3;
        texDesc.MipLevels = 1;
        texDesc.ArraySize = 1;
        texDesc.Format = DXGI_FORMAT_R32_FLOAT;
        texDesc.SampleDesc.Count = 1;
        texDesc.Usage = D3D11_USAGE_DEFAULT;
        texDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        texDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
        
        pImpl->m_device->CreateTexture2D(&texDesc, nullptr, &pImpl->m_modelInputTexture);
        pImpl->m_device->CreateUnorderedAccessView(pImpl->m_modelInputTexture.Get(), nullptr, &pImpl->m_modelInputUAV);
        
        texDesc.MiscFlags = 0;
        pImpl->m_device->CreateTexture2D(&texDesc, nullptr, &pImpl->m_modelOutputTexture);
        pImpl->m_device->CreateShaderResourceView(pImpl->m_modelOutputTexture.Get(), nullptr, &pImpl->m_modelOutputSRV);

        ComPtr<IDXGIResource1> dxgiResource;
        HANDLE sharedHandle;
        pImpl->m_modelInputTexture.As(&dxgiResource);
        dxgiResource->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &sharedHandle);
        pImpl->m_onnx->GetDMLHelper().GetD3D12Device()->OpenSharedHandle(sharedHandle, IID_PPV_ARGS(&pImpl->m_modelInputD3D12));
        CloseHandle(sharedHandle);

        pImpl->m_modelOutputTexture.As(&dxgiResource);
        dxgiResource->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &sharedHandle);
        pImpl->m_onnx->GetDMLHelper().GetD3D12Device()->OpenSharedHandle(sharedHandle, IID_PPV_ARGS(&pImpl->m_modelOutputD3D12));
        CloseHandle(sharedHandle);
        
        auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(128 * 128 * 3 * sizeof(float));
        auto defaultHeap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        pImpl->m_onnx->GetDMLHelper().GetD3D12Device()->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &bufferDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&pImpl->m_privateModelInputBuffer));
        pImpl->m_onnx->GetDMLHelper().GetD3D12Device()->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &bufferDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&pImpl->m_privateModelOutputBuffer));

        pImpl->m_device5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&pImpl->m_d3d11Fence));
        pImpl->m_d3d11Fence->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &sharedHandle);
        pImpl->m_onnx->GetDMLHelper().GetD3D12Device()->OpenSharedHandle(sharedHandle, IID_PPV_ARGS(&pImpl->m_d3d12Fence));
        CloseHandle(sharedHandle);

        const UINT mask_size = 128;
        D3D11_TEXTURE2D_DESC maskDesc = {};
        maskDesc.Width = mask_size;
        maskDesc.Height = mask_size;
        maskDesc.MipLevels = 1;
        maskDesc.ArraySize = 1;
        maskDesc.Format = DXGI_FORMAT_R8_UNORM;
        maskDesc.SampleDesc.Count = 1;
        maskDesc.Usage = D3D11_USAGE_IMMUTABLE;
        maskDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        std::vector<uint8_t> full_mask(mask_size * mask_size, 255);
        D3D11_SUBRESOURCE_DATA subData = { full_mask.data(), mask_size * sizeof(uint8_t), 0 };
        pImpl->m_device->CreateTexture2D(&maskDesc, &subData, &pImpl->m_mask_tex);
        pImpl->m_device->CreateShaderResourceView(pImpl->m_mask_tex.Get(), nullptr, &pImpl->m_mask_srv);
        
        ComPtr<ID3D11Resource> mainFrameResource;
        frameData.pInputSRV->GetResource(&mainFrameResource);
        ComPtr<ID3D11Texture2D> mainFrameTexture;
        mainFrameResource.As(&mainFrameTexture);
        D3D11_TEXTURE2D_DESC frameDesc;
        mainFrameTexture->GetDesc(&frameDesc);
        frameDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        pImpl->m_device->CreateTexture2D(&frameDesc, nullptr, &pImpl->m_generatedSwappedFaceTexture);
        pImpl->m_generatedSwappedFaceTexture->SetPrivateData(WKPDID_D3DDebugObjectName, sizeof("GeneratedSwappedFaceTexture") - 1, "GeneratedSwappedFaceTexture");
        pImpl->m_device->CreateRenderTargetView(pImpl->m_generatedSwappedFaceTexture.Get(), nullptr, &pImpl->m_generatedSwappedFaceRTV);
        pImpl->m_device->CreateShaderResourceView(pImpl->m_generatedSwappedFaceTexture.Get(), nullptr, &pImpl->m_generatedSwappedFaceSRV);
        pImpl->m_device->CreateUnorderedAccessView(pImpl->m_generatedSwappedFaceTexture.Get(), nullptr, &pImpl->m_generatedSwappedFaceUAV);
    }
    
    if (!pImpl->hasSourceFace || frameData.faces.empty()) {
        if (pImpl->m_generatedSwappedFaceTexture) {
             ComPtr<ID3D11Resource> srcResource;
             frameData.pInputSRV->GetResource(&srcResource);
             pImpl->m_context->CopyResource(pImpl->m_generatedSwappedFaceTexture.Get(), srcResource.Get());
        }
        return;
    }

    ComPtr<ID3D11Resource> srcResource;
    frameData.pInputSRV->GetResource(&srcResource);
    pImpl->m_context->CopyResource(pImpl->m_generatedSwappedFaceTexture.Get(), srcResource.Get());
    
    D3D11_TEXTURE2D_DESC finalFrameDesc;
    pImpl->m_generatedSwappedFaceTexture->GetDesc(&finalFrameDesc);

    for (const auto& target_face : frameData.faces) {
        Eigen::Matrix<double, 5, 2> arcface_dst_pts;
        arcface_dst_pts << 38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366, 41.5493, 92.3655, 70.7299, 92.2041;
        Eigen::Matrix<double, 2, 3> M_align_forward = WinTegrity::estimateSimilarityTransform(target_face.landmarks, arcface_dst_pts);
        Eigen::Matrix<double, 2, 3> M_align_inverse = WinTegrity::invertAffineTransform(M_align_forward);
        
        TransformConstants warp_constants{};
        warp_constants.inverseTransformMatrix._11 = (float)M_align_inverse(0,0);
        warp_constants.inverseTransformMatrix._12 = (float)M_align_inverse(0,1);
        warp_constants.inverseTransformMatrix._13 = (float)M_align_inverse(0,2);
        warp_constants.inverseTransformMatrix._21 = (float)M_align_inverse(1,0);
        warp_constants.inverseTransformMatrix._22 = (float)M_align_inverse(1,1);
        warp_constants.inverseTransformMatrix._23 = (float)M_align_inverse(1,2);
        
        pImpl->m_warp_affine->Execute(pImpl->m_context.Get(), warp_constants, frameData.pInputSRV, pImpl->m_modelInputUAV.Get());
        
        pImpl->m_context->Flush();

        pImpl->m_d3d12CmdAllocator->Reset();
        pImpl->m_d3d12CmdList->Reset(pImpl->m_d3d12CmdAllocator.Get(), nullptr);

        CD3DX12_RESOURCE_BARRIER preCopyBarriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(pImpl->m_modelInputD3D12.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
            CD3DX12_RESOURCE_BARRIER::Transition(pImpl->m_privateModelInputBuffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST)
        };
        pImpl->m_d3d12CmdList->ResourceBarrier(_countof(preCopyBarriers), preCopyBarriers);
        
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout = {};
        D3D12_RESOURCE_DESC modelInputDesc = pImpl->m_modelInputD3D12->GetDesc();
        pImpl->m_onnx->GetDMLHelper().GetD3D12Device()->GetCopyableFootprints(&modelInputDesc, 0, 1, 0, &layout, nullptr, nullptr, nullptr);
        CD3DX12_TEXTURE_COPY_LOCATION Dst(pImpl->m_privateModelInputBuffer.Get(), layout);
        CD3DX12_TEXTURE_COPY_LOCATION Src(pImpl->m_modelInputD3D12.Get(), 0);
        pImpl->m_d3d12CmdList->CopyTextureRegion(&Dst, 0, 0, 0, &Src, nullptr);
        
        CD3DX12_RESOURCE_BARRIER postCopyBarriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(pImpl->m_modelInputD3D12.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
            CD3DX12_RESOURCE_BARRIER::Transition(pImpl->m_privateModelInputBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)
        };
        pImpl->m_d3d12CmdList->ResourceBarrier(_countof(postCopyBarriers), postCopyBarriers);
        pImpl->m_d3d12CmdList->Close();
        ID3D12CommandList* lists_pre[] = { pImpl->m_d3d12CmdList.Get() };
        pImpl->m_onnx->GetDMLHelper().GetCommandQueue()->ExecuteCommandLists(1, lists_pre);
        
        pImpl->m_onnx->GetDMLHelper().WaitForGpuIdle();
        
        try {
            Eigen::Map<const Eigen::Matrix<float, 1, 512>> source_embedding_map(pImpl->sourceEmbedding.data());
            Eigen::Matrix<float, 1, 512> latent = source_embedding_map * pImpl->emap;
            latent.normalize();
            std::vector<float> latent_data(latent.data(), latent.data() + latent.size());
            auto source_tensor = pImpl->m_onnx->CreateCpuTensor<float>(latent_data, {1, 512});
            
            pImpl->m_d3d12CmdAllocator->Reset();
            pImpl->m_d3d12CmdList->Reset(pImpl->m_d3d12CmdAllocator.Get(), nullptr);

            CD3DX12_RESOURCE_BARRIER preOnnxBarriers[] = {
                CD3DX12_RESOURCE_BARRIER::Transition(pImpl->m_privateModelInputBuffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                CD3DX12_RESOURCE_BARRIER::Transition(pImpl->m_privateModelOutputBuffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            };
            pImpl->m_d3d12CmdList->ResourceBarrier(_countof(preOnnxBarriers), preOnnxBarriers);
            pImpl->m_d3d12CmdList->Close();
            ID3D12CommandList* lists_pre_onnx[] = { pImpl->m_d3d12CmdList.Get() };
            pImpl->m_onnx->GetDMLHelper().GetCommandQueue()->ExecuteCommandLists(1, lists_pre_onnx);

            auto target_tensor = pImpl->m_onnx->CreateTensorFromD3DResource(pImpl->m_privateModelInputBuffer.Get(), {1, 3, 128, 128}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(std::move(target_tensor));
            input_tensors.push_back(std::move(source_tensor));
            const char* input_names[] = {"target", "source"};
            const char* output_names[] = {"output"};
            auto output_tensor = pImpl->m_onnx->CreateTensorFromD3DResource(pImpl->m_privateModelOutputBuffer.Get(), {1, 3, 128, 128}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
            Ort::RunOptions run_options;
            pImpl->swap_session->Run(run_options, input_names, input_tensors.data(), input_tensors.size(), output_names, &output_tensor, 1);
            
            pImpl->m_d3d12CmdAllocator->Reset();
            pImpl->m_d3d12CmdList->Reset(pImpl->m_d3d12CmdAllocator.Get(), nullptr);
            CD3DX12_RESOURCE_BARRIER postOnnxBarriers[] = {
                 CD3DX12_RESOURCE_BARRIER::Transition(pImpl->m_privateModelInputBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON),
                 CD3DX12_RESOURCE_BARRIER::Transition(pImpl->m_privateModelOutputBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON)
            };
            pImpl->m_d3d12CmdList->ResourceBarrier(_countof(postOnnxBarriers), postOnnxBarriers);
            pImpl->m_d3d12CmdList->Close();
            ID3D12CommandList* lists_post_onnx[] = { pImpl->m_d3d12CmdList.Get() };
            pImpl->m_onnx->GetDMLHelper().GetCommandQueue()->ExecuteCommandLists(1, lists_post_onnx);
        }
        catch (const Ort::Exception& e) { 
             if (frameData.pConsole) frameData.pConsole->AddLog("[ERROR] ONNX Runtime Exception: %s", e.what());
             continue; 
        }

        pImpl->m_d3d12CmdAllocator->Reset();
        pImpl->m_d3d12CmdList->Reset(pImpl->m_d3d12CmdAllocator.Get(), nullptr);
        
        CD3DX12_RESOURCE_BARRIER preCopyBarriers2[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(pImpl->m_modelOutputD3D12.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
            CD3DX12_RESOURCE_BARRIER::Transition(pImpl->m_privateModelOutputBuffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)
        };
        pImpl->m_d3d12CmdList->ResourceBarrier(_countof(preCopyBarriers2), preCopyBarriers2);

        D3D12_RESOURCE_DESC modelOutputDesc = pImpl->m_modelOutputD3D12->GetDesc();
        pImpl->m_onnx->GetDMLHelper().GetD3D12Device()->GetCopyableFootprints(&modelOutputDesc, 0, 1, 0, &layout, nullptr, nullptr, nullptr);
        CD3DX12_TEXTURE_COPY_LOCATION Dst2(pImpl->m_modelOutputD3D12.Get(), 0);
        CD3DX12_TEXTURE_COPY_LOCATION Src2(pImpl->m_privateModelOutputBuffer.Get(), layout);
        pImpl->m_d3d12CmdList->CopyTextureRegion(&Dst2, 0, 0, 0, &Src2, nullptr);

        CD3DX12_RESOURCE_BARRIER postCopyBarriers2[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(pImpl->m_modelOutputD3D12.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON),
            CD3DX12_RESOURCE_BARRIER::Transition(pImpl->m_privateModelOutputBuffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)
        };
        pImpl->m_d3d12CmdList->ResourceBarrier(_countof(postCopyBarriers2), postCopyBarriers2);
        pImpl->m_d3d12CmdList->Close();
        ID3D12CommandList* lists_post[] = { pImpl->m_d3d12CmdList.Get() };
        pImpl->m_onnx->GetDMLHelper().GetCommandQueue()->ExecuteCommandLists(1, lists_post);
        
        pImpl->m_onnx->GetDMLHelper().GetCommandQueue()->Signal(pImpl->m_d3d12Fence.Get(), ++pImpl->m_fenceValue);
        pImpl->m_context4->Wait(pImpl->m_d3d11Fence.Get(), pImpl->m_fenceValue);

        TransformConstants blend_constants{};
        blend_constants.inverseTransformMatrix._11 = (float)M_align_inverse(0,0);
        blend_constants.inverseTransformMatrix._12 = (float)M_align_inverse(0,1);
        blend_constants.inverseTransformMatrix._13 = (float)M_align_inverse(0,2);
        blend_constants.inverseTransformMatrix._21 = (float)M_align_inverse(1,0);
        blend_constants.inverseTransformMatrix._22 = (float)M_align_inverse(1,1);
        blend_constants.inverseTransformMatrix._23 = (float)M_align_inverse(1,2);
        
        pImpl->m_blend->Execute(pImpl->m_context.Get(), blend_constants, pImpl->m_modelOutputSRV.Get(), pImpl->m_mask_srv.Get(), pImpl->m_generatedSwappedFaceSRV.Get(), pImpl->m_generatedSwappedFaceUAV.Get(), finalFrameDesc.Width, finalFrameDesc.Height);
    }
}

ID3D12Resource* FaceSwap::GetPrivateInputBuffer() const
{
    return pImpl->m_privateModelInputBuffer.Get();
}

ID3D12Resource* FaceSwap::GetPrivateOutputBuffer() const
{
    return pImpl->m_privateModelOutputBuffer.Get();
}

ID3D12Fence* FaceSwap::GetCompletionFence() const
{
    return pImpl->m_d3d12Fence.Get();
}

UINT64 FaceSwap::GetLastFenceValue() const
{
    return pImpl->m_fenceValue;
}

ID3D11Texture2D* FaceSwap::GetModelInputTexture() const
{
    return pImpl->m_modelInputTexture.Get();
}

}