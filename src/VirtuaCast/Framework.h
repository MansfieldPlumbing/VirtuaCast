// File: src/VirtuaCast/Framework.h

#pragma once

#include <d3d11.h>
#include <vector>
#include <memory>
#include <wrl/client.h>
#include <Eigen/Dense>

namespace VirtuaCast {

    class OnnxRuntime;
    class Console;

    struct FaceResult {
        float bbox[4];
        Eigen::Matrix<double, 5, 2> landmarks;
        float detection_score;
        std::vector<float> embedding;
    };

    struct FrameData {
        ID3D11Device* pDevice = nullptr;
        ID3D11DeviceContext* pContext = nullptr;
        ID3D11ShaderResourceView* pInputSRV = nullptr;
        ID3D11RenderTargetView* pOutputRTV = nullptr;
        Console* pConsole = nullptr;
        std::vector<FaceResult> faces;
    };

    class VisionModule {
    public:
        virtual ~VisionModule() = default;
        virtual const char* GetName() const = 0;
        // MODIFIED: This is the definitive signature that all derived modules must now implement.
        virtual HRESULT Initialize(VirtuaCast::OnnxRuntime& onnx, const std::string& model_dir, Console* console) = 0;
        virtual void Process(FrameData& frameData) = 0;
    };

} // namespace VirtuaCast