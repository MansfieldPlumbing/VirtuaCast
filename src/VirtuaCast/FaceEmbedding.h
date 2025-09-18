#pragma once
#include "Framework.h"
#include <memory>
#include <string>

namespace Ort {
    struct Env;
    struct Session;
    struct SessionOptions;
    struct MemoryInfo;
}

namespace VirtuaCast {

    class OnnxRuntime;

    class FaceEmbedder : public VisionModule {
    public:
        FaceEmbedder();
        ~FaceEmbedder();

        const char* GetName() const override { return "Face Embedder"; }
        // MODIFIED: Signature updated to match base class.
        HRESULT Initialize(VirtuaCast::OnnxRuntime& onnx, const std::string& model_dir, Console* console) override;
        void Process(FrameData& frameData) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };

}