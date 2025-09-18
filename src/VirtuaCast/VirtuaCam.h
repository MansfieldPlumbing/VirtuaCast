// File: src/VirtuaCast/VirtuaCast.h

#pragma once

#include <windows.h>
#include <d3d11_4.h>
#include <wrl/client.h>
#include <memory>
#include <string>
#include <vector>

#include "KeyCommands.h"
#include "DebugOutput.h"

namespace VirtuaCast {
    class SwapChain;
    class Consumer;
    class OnnxRuntime;
    class Generator;
    class CameraSource;
    class Console;
    class BoundingBoxRenderer;
    class UI;
    class KeyCommandHandler;
    class FaceDetector;
    class FaceEmbedder;
    class FaceSwap;
    class SourceFaceManager;
    class Producer;
    enum class ActiveSource;

    class Application {
    public:
        Application(const wchar_t* title, int width, int height);
        ~Application();
        void Run();

        FaceDetector* GetFaceDetector() { return m_face_detector_module.get(); }
        FaceEmbedder* GetFaceEmbedder() { return m_face_embedder_module.get(); }
        FaceSwap* GetFaceSwap() { return m_face_swap.get(); }
        OnnxRuntime* GetOnnxRuntime() { return m_onnx_runtime.get(); }
        ID3D11Device* GetDevice() { return m_device.Get(); }
        ID3D11DeviceContext* GetContext() { return m_context.Get(); }
        ID3D11ShaderResourceView* GetFinalFrameSRV() { return m_finalFrameSRV.Get(); }
        Console* GetConsole() { return m_console.get(); }

    private:
        void Initialize();
        void Teardown();
        void MainLoop();
        void Tick();
        void SetupUITheme();
        void CreateFinalFrameResources();
        void ToggleFullscreen();

        static LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
        static Application* GetThisFromHandle(HWND hWnd) {
            return reinterpret_cast<Application*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));
        }

        HWND m_hwnd = nullptr;
        std::wstring m_title;
        int m_width, m_height;
        Microsoft::WRL::ComPtr<ID3D11Device> m_device;
        Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_context;
        
        Microsoft::WRL::ComPtr<ID3D11Fence> m_frameCompletionFence;
        UINT64 m_frameFenceValue = 0;
        HANDLE m_fenceEvent = nullptr;

        Microsoft::WRL::ComPtr<ID3D11VertexShader> m_stretchVS;
        Microsoft::WRL::ComPtr<ID3D11PixelShader> m_stretchPS;
        Microsoft::WRL::ComPtr<ID3D11SamplerState> m_linearSampler;

        Microsoft::WRL::ComPtr<ID3D11Texture2D> m_finalFrameTexture;
        Microsoft::WRL::ComPtr<ID3D11RenderTargetView> m_finalFrameRTV;
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_finalFrameSRV;
        
        Microsoft::WRL::ComPtr<ID3D11Texture2D> m_pipelineInputTexture;
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_pipelineInputSRV;
        
        UINT m_frame_width = 0, m_frame_height = 0;

        std::unique_ptr<SwapChain> m_swapChain;
        std::unique_ptr<Consumer> m_consumer;
        std::unique_ptr<OnnxRuntime> m_onnx_runtime;
        std::unique_ptr<Generator> m_generator;
        std::unique_ptr<CameraSource> m_camera_source;
        std::unique_ptr<Console> m_console;
        std::unique_ptr<BoundingBoxRenderer> m_bbox_renderer;
        std::unique_ptr<UI> m_ui;
        std::unique_ptr<KeyCommandHandler> m_key_handler;
        std::unique_ptr<Producer> m_producer;
        
        std::unique_ptr<FaceDetector> m_face_detector_module;
        std::unique_ptr<FaceEmbedder> m_face_embedder_module;
        std::unique_ptr<FaceSwap> m_face_swap;
        std::unique_ptr<SourceFaceManager> m_source_face_manager;

        AppState m_app_state;
        
        UINT_PTR m_discoveryTimer = 0;
        ActiveSource m_active_source;
        int m_active_stream_index = -1;
        int m_active_camera_index = -1;
        bool m_mirror_preview = false;
        
        std::vector<std::pair<UINT, UINT>> m_resolutions;
        int m_current_res_index = -1;
    };
}