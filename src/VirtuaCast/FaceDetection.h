// File: src/VirtuaCast/FaceDetection.h

#pragma once
#include "Framework.h"
#include <memory>
#include <string>

// Forward-declare to keep ONNX headers out of this public interface
namespace Ort {
    struct Session;
}

namespace VirtuaCast {

    class FaceDetector : public VisionModule {
    public:
        FaceDetector();
        ~FaceDetector();

        const char* GetName() const override { return "Face Detector"; }
        // MODIFIED: Signature updated to match base class.
        HRESULT Initialize(VirtuaCast::OnnxRuntime& onnx, const std::string& model_dir, Console* console) override;
        void Process(FrameData& frameData) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };
}