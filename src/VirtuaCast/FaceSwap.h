#pragma once
#include "Framework.h"
#include <memory>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <d3d12.h>
#include <d3d11.h>

namespace VirtuaCast {

    class OnnxRuntime;
    class WarpAffine;
    class Blend;
    class Console;

    class FaceSwap : public VisionModule {
    public:
        FaceSwap();
        ~FaceSwap();

        void SetSourceFace(const std::vector<float>& embedding);
        void ClearSourceFace();
        
        ID3D11ShaderResourceView* GetGeneratedSwappedFaceSRV() const;
        ID3D11Texture2D* GetGeneratedSwappedFaceTexture() const;
        ID3D11ShaderResourceView* GetModelOutputSRV() const;
        ID3D11ShaderResourceView* GetMaskSRV() const;

        const char* GetName() const override { return "Face Swap"; }
        HRESULT Initialize(VirtuaCast::OnnxRuntime& onnx, const std::string& model_dir, Console* console) override;
        void Process(FrameData& frameData) override;
        
        ID3D12Resource* GetPrivateInputBuffer() const;
        ID3D12Resource* GetPrivateOutputBuffer() const;
        ID3D12Fence* GetCompletionFence() const;
        UINT64 GetLastFenceValue() const;
        ID3D11Texture2D* GetModelInputTexture() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };

}