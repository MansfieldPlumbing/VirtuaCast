// File: src/VirtuaCast/SourceFace.h

#pragma once

#include "Framework.h"
#include <string>
#include <vector>
#include <memory>

namespace VirtuaCast {

    class OnnxRuntime;
    class FaceDetector;
    class FaceEmbedder;
    class FaceSwap;
    class Console;

    struct SourceFace {
        std::wstring image_path;
        std::string display_name;
        std::vector<float> embedding;
    };

    class SourceFaceManager {
    public:
        SourceFaceManager();
        ~SourceFaceManager();

        HRESULT Initialize(
            const std::string& sources_directory,
            OnnxRuntime& onnx,
            FaceDetector& detector,
            FaceEmbedder& embedder,
            Console* console
        );

        void CycleSource(int delta);
        const SourceFace* GetCurrentSource() const;
        void SetCurrentSourceAsActive(FaceSwap& swapper);

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };
}