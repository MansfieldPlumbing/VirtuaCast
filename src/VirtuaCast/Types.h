// File: src/VirtuaCast/Types.h

#pragma once

#include <cstdint>

// This namespace holds C-style POD (Plain Old Data) structs ported from legacy
// code. They are used by the CPU-based image processing algorithms for compatibility.
namespace WinTegrity {

    // A C-style struct for passing raw, CPU-side image data.
    struct TegrityImageBuffer {
        unsigned char* data;
        int width;
        int height;
        int channels;
    };

    // A C-style struct for defining a simple rectangle.
    struct TegrityRect {
        int x;
        int y;
        int width;
        int height;
    };

    // Configuration struct for the ML pipeline.
    struct TegrityPipelineConfig {
        int gpu_device_id = 0;
        int target_alignment_size = 128;
        float mask_blur_intensity = 5.0f;
        float mask_erosion_level = 3.0f;
        uint8_t mask_threshold = 10;
        bool enable_color_correction = true;
    };
    
    // C-style equivalent of the modern FaceResult, used by older algorithms.
    struct TegrityDetectedFace {
        float bbox[4];
        double kps[5][2];
        float det_score;
    };
}