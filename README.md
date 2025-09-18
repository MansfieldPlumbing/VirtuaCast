# VirtuaCast
VirtuaCast: A high-performance C++ pipeline for real-time AI vision. Features a modern producer/consumer architecture using DirectX fences for robust, zero-copy GPU texture sharing. Integrated with a custom virtual camera, it turns any processed stream into a universal video source for apps like OBS and Zoom.

# VirtuaCast: The Engine for Real-Time AI Video

![VirtuaCast Demo GIF](https://your-link-to-a-demo.gif)
*(**Action:** TO BE INCLDED "VirtuaCam" device.)*

---

## What is VirtuaCast?

VirtuaCast is a native C++ platform for building high-performance, real-time video processing and AI inference pipelines on Windows. It is the professional-grade evolution of my Python-based `FaceOn Studio` prototype, rebuilt from the ground up to solve the core engineering challenges of GPU-accelerated applications.

It's not just an application; it's an engine designed to be the foundational "GPU plumbing" for any project that needs to ingest, process, and broadcast video streams with minimal latency and maximum control.

---

## Key Features

*   **🚀 Native C++ Performance:** The entire pipeline, from camera ingestion to AI inference and final compositing, is written in modern C++ for maximum performance, eliminating bottlenecks like the Python GIL.
*   **🔗 Modern Producer/Consumer Architecture:** A technically superior successor to established protocols like Spout. It uses **DirectX Fences** for explicit, robust GPU synchronization between processes, eliminating a major pain point of traditional texture sharing.
*   **📸 Universal Virtual Camera Output:** VirtuaCast's output is broadcast via a custom-built `VirtuaCam` driver. This makes any processed video stream appear as a standard webcam, allowing it to be used as a source in **any application on Windows** (OBS, Zoom, Discord, Teams, etc.) without requiring any plugins.
*   **🧠 GPU-Accelerated AI:** Features a deep integration with ONNX Runtime using the **DirectML execution provider**, allowing complex AI models for face detection, embedding, and swapping to run directly on the GPU.
*   **🤖 Modular by Design:** Architected with a clear separation of concerns (`CameraSource`, `Consumer`, `FaceDetector`, `FaceSwap`, `Producer`), making the engine easy to maintain and extend.

---

## Architectural Deep Dive

VirtuaCast is built as a series of interconnected modules that manage the flow of GPU resources:

1.  **Input:** The `CameraSource` or `Consumer` module ingests a video stream. The `Consumer` uses a `BroadcastManifest` to discover running producers and establishes a connection using shared texture handles and GPU fences.
2.  **Processing:** The frame is processed through a series of `VisionModule`s. This is where custom HLSL compute shaders (like `PreprocessInswapper.hlsl` and `PostprocessBlend.hlsl`) and ONNX models are executed.
3.  **Output:** The final composited texture is handed to the `Producer` module, which signals its own GPU fence and updates its manifest.
4.  **Broadcast:** The `VirtuaCam` driver consumes the `Producer`'s stream, making it available system-wide.

---

## Project Status

This project is an active work-in-progress, representing my ongoing deep dive into systems-level graphics programming.

-   [x] Core D3D11/D3D12 Infrastructure
-   [x] Modern Producer/Consumer Architecture with Fence Synchronization
-   [x] ONNX Runtime Integration with DirectML
-   [x] Modular Vision Pipeline (`FaceDetector`, `FaceEmbedder`, `FaceSwap`)
-   [ ] **Next Up:** Finalizing the `VirtuaCam` DLL integration and building a more robust UI with ImGui.

---

### Tech Stack

*   **Core:** C++17
*   **Graphics API:** DirectX 11 / DirectX 12
*   **Windows APIs:** Windows Media Foundation (WMF), COM
*   **AI Inference:** ONNX Runtime (with DirectML Execution Provider)
*   **Shaders:** HLSL (Compute & Pixel Shaders)
*   **UI:** Dear ImGui
*   **Dependencies:** Eigen (for math), WIL (for helpers)
