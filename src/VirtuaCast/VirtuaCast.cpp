#define WIN32_LEAN_AND_MEAN
#include "VirtuaCast.h"
#include <windows.h>
#include <d3d11_4.h>
#include <dwmapi.h>
#include <stdexcept>
#include <string>
#include <memory>
#include <d3dcompiler.h>
#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"
#include "Framework.h"
#include "KeyCommands.h"
#include "SwapChain.h"
#include "Consumer.h"
#include "OnnxRuntime.h"
#include "UI.h"
#include "Generator.h"
#include "CameraSource.h"
#include "FaceDetection.h"
#include "FaceEmbedding.h"
#include "FaceSwap.h"
#include "Console.h"
#include "BoundingBox.h"
#include "resource.h"
#include "CameraShader.h"
#include "SourceFace.h"
#include "DebugOutput.h"
#include "Producer.h"
#include "DataBus.h"
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dwmapi.lib")
#pragma comment(lib, "uxtheme.lib")
#pragma comment(lib, "d3dcompiler.lib")
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
using namespace Microsoft::WRL;
int APIENTRY wWinMain(HINSTANCE hInstance, HINSTANCE, LPWSTR, int nCmdShow)
{
try
{
VirtuaCast::Application app(L"VirtuaCast", 1280, 800);
app.Run();
}
catch (const std::exception &e)
{
MessageBoxA(NULL, e.what(), "Fatal Error", MB_OK | MB_ICONERROR);
return 1;
}
return 0;
}
namespace VirtuaCast
{
Application::Application(const wchar_t *title, int width, int height)
    : m_title(title), m_width(width), m_height(height), m_active_source(ActiveSource::Generator)
{
    m_resolutions = {{640, 480}, {1280, 720}, {1920, 1080}};
    m_current_res_index = 1;
}
Application::~Application() { Teardown(); }

void Application::Run()
{
    Initialize();
    MainLoop();
}

void Application::Initialize()
{
    WNDCLASSEXW wc = {sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(NULL), LoadIcon(GetModuleHandle(NULL), MAKEINTRESOURCE(IDI_ICON1)), NULL, NULL, NULL, m_title.c_str(), NULL};
    RegisterClassExW(&wc);
    m_hwnd = CreateWindowW(wc.lpszClassName, m_title.c_str(), WS_OVERLAPPEDWINDOW, 100, 100, m_width, m_height, NULL, NULL, wc.hInstance, this);

    m_swapChain = std::make_unique<SwapChain>();
    if (FAILED(m_swapChain->Initialize(m_hwnd, m_width, m_height, &m_device, &m_context)))
    {
        throw std::runtime_error("Failed to initialize SwapChain and D3D11 device.");
    }

    if (FAILED(m_device->CreateDeferredContext(0, &m_pipelineContext)))
    {
        throw std::runtime_error("Failed to create deferred context for pipeline.");
    }

    ComPtr<ID3D11Device5> device5;
    if (FAILED(m_device.As(&device5)))
    {
        throw std::runtime_error("D3D11.5 Device interface not supported.");
    }
    if (FAILED(device5->CreateFence(0, D3D11_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_frameCompletionFence))))
    {
         throw std::runtime_error("Failed to create D3D11 fence.");
    }
    m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (m_fenceEvent == nullptr)
    {
        throw std::runtime_error("Failed to create fence event.");
    }

    m_console = std::make_unique<Console>();
    m_console->AddLog("VirtuaCast console initialized. Press `~` to toggle.");

    m_onnx_runtime = std::make_unique<OnnxRuntime>();
    m_onnx_runtime->Initialize(m_device.Get(), m_console.get());

    m_consumer = std::make_unique<Consumer>();
    m_consumer->Initialize(m_device.Get(), m_context.Get());
    m_consumer->DiscoverStreams();
    m_discoveryTimer = SetTimer(m_hwnd, 1, 2000, nullptr);

    m_camera_source = std::make_unique<CameraSource>();
    m_camera_source->Initialize(m_device.Get());

    m_generator = std::make_unique<Generator>();
    m_generator->Initialize(m_device.Get());
    
    m_face_detector_module = std::make_unique<FaceDetector>();
    m_face_detector_module->Initialize(*m_onnx_runtime, "models", m_console.get());
    m_face_embedder_module = std::make_unique<FaceEmbedder>();
    m_face_embedder_module->Initialize(*m_onnx_runtime, "models", m_console.get());
    m_face_swap = std::make_unique<FaceSwap>();
    m_face_swap->Initialize(*m_onnx_runtime, "models", m_console.get());

    m_source_face_manager = std::make_unique<SourceFaceManager>();
    m_source_face_manager->Initialize("Sources", *m_onnx_runtime, *m_face_detector_module, *m_face_embedder_module, m_console.get());

    m_bbox_renderer = std::make_unique<BoundingBoxRenderer>();
    m_bbox_renderer->Initialize(m_device.Get());

    m_ui = std::make_unique<UI>();
    m_key_handler = std::make_unique<KeyCommandHandler>();

    CreateFinalFrameResources();

    m_dataBus = std::make_unique<DataBus>();
    m_dataBus->Initialize(m_device.Get(), m_context.Get(), m_frame_width, m_frame_height);

    m_producer = std::make_unique<Producer>();
    if(FAILED(m_producer->Initialize(m_device.Get(), m_context.Get(), m_finalFrameTexture.Get())))
    {
        m_console->AddLog("[ERROR] Failed to initialize Producer module. Frame sharing will be disabled.");
        m_producer.reset();
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    SetupUITheme();
    ImGui_ImplWin32_Init(m_hwnd);
    ImGui_ImplDX11_Init(m_device.Get(), m_context.Get());

    ShowWindow(m_hwnd, SW_SHOWDEFAULT);
    UpdateWindow(m_hwnd);
}

void Application::Teardown()
{
    if (m_fenceEvent) CloseHandle(m_fenceEvent);
    if (m_discoveryTimer)
        KillTimer(m_hwnd, m_discoveryTimer);
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    m_producer.reset();
    m_dataBus.reset();
    m_bbox_renderer.reset();
    m_source_face_manager.reset();
    m_face_swap.reset();
    m_face_embedder_module.reset();
    m_face_detector_module.reset();
    m_generator.reset();
    m_camera_source.reset();
    m_consumer.reset();
    m_onnx_runtime.reset();
    m_swapChain.reset();
    m_pipelineContext.Reset();
    m_context.Reset();
    m_device.Reset();

    if (m_hwnd)
        DestroyWindow(m_hwnd);
    UnregisterClassW(m_title.c_str(), GetModuleHandle(NULL));
}

void Application::MainLoop()
{
    MSG msg;
    ZeroMemory(&msg, sizeof(msg));
    while (msg.message != WM_QUIT)
    {
        if (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            continue;
        }
        Tick();
    }
}

void Application::ToggleFullscreen()
{
    if (m_swapChain)
    {
        m_swapChain->ToggleFullscreen();
        m_console->AddLog("Toggled fullscreen mode.");
    }
}

void Application::Tick()
{
    m_key_handler->ProcessInput(m_app_state);
    if (m_app_state.should_quit)
    {
        PostMessage(m_hwnd, WM_CLOSE, 0, 0);
        return;
    }
    if (m_app_state.trigger_toggle_console)
        m_console->Toggle();
    if (m_app_state.trigger_toggle_fullscreen)
        ToggleFullscreen();
    if (m_app_state.source_face_change_delta != 0)
    {
        m_source_face_manager->CycleSource(m_app_state.source_face_change_delta);
        m_source_face_manager->SetCurrentSourceAsActive(*m_face_swap);
    }
    if (m_app_state.trigger_run_debug_trace)
    {
        const SourceFace *current_source = m_source_face_manager->GetCurrentSource();
        if (current_source)
        {
            m_context->Flush();
            TegritySourceFace debug_face;
            debug_face.name = current_source->display_name.c_str();
            debug_face.normed_embedding = current_source->embedding.data();
            TegrityDebug_RunFullSwapTrace(this, &debug_face);
        }
        else
        {
            m_console->AddLog("[Debug] Cannot run trace: No source face is selected.");
        }
    }

    if (m_app_state.resolution_change_delta != 0)
    {
        if (m_active_source == ActiveSource::Camera && m_camera_source->IsActive())
        {
            m_current_res_index += m_app_state.resolution_change_delta;
            if (m_current_res_index < 0)
                m_current_res_index = (int)m_resolutions.size() - 1;
            if (m_current_res_index >= (int)m_resolutions.size())
                m_current_res_index = 0;

            UINT w = m_resolutions[m_current_res_index].first;
            UINT h = m_resolutions[m_current_res_index].second;
            m_camera_source->ActivateCamera(m_active_camera_index, w, h);
            m_console->AddLog("Set camera resolution to %dx%d", w, h);
        }
    }

    ImGui_ImplDX11_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    ActiveSource last_source = m_active_source;
    int last_cam_idx = m_active_camera_index;
    int last_stream_idx = m_active_stream_index;

    ID3D11ShaderResourceView *previewSRV = m_finalFrameSRV.Get();
    if (m_app_state.active_mode == ProcessingMode::Swap)
    {
        previewSRV = m_face_swap->GetGeneratedSwappedFaceSRV();
    }

    m_ui->Render(m_app_state, *m_consumer, *m_camera_source, *m_source_face_manager, m_active_stream_index, m_active_camera_index, m_active_source, previewSRV, m_frame_width, m_frame_height);

    if (last_source != m_active_source || last_cam_idx != m_active_camera_index || last_stream_idx != m_active_stream_index)
    {
        m_consumer->Disconnect();
        m_camera_source->DeactivateCamera();
    }

    ID3D11ShaderResourceView *activeSRV = nullptr;

    if (m_active_source == ActiveSource::SharedStream)
    {
        const auto &streams = m_consumer->GetDiscoveredStreams();
        if (m_active_stream_index >= 0 && m_active_stream_index < (int)streams.size())
        {
            if (!m_consumer->IsConnected())
            {
                m_consumer->Connect(streams[m_active_stream_index]);
                m_console->AddLog("Connecting to shared stream: %S", streams[m_active_stream_index].processName.c_str());
            }
            activeSRV = m_consumer->UpdateAndGetSRV();
        }
    }
    else if (m_active_source == ActiveSource::Camera)
    {
        if (m_active_camera_index >= 0 && m_active_camera_index < (int)m_camera_source->GetAvailableCameras().size())
        {
            if (!m_camera_source->IsActive())
            {
                UINT w = m_resolutions[m_current_res_index].first;
                UINT h = m_resolutions[m_current_res_index].second;
                m_camera_source->ActivateCamera(m_active_camera_index, w, h);
                m_console->AddLog("Activating camera: %S at %dx%d", m_camera_source->GetAvailableCameras()[m_active_camera_index].name.c_str(), w, h);
            }
            activeSRV = m_camera_source->GetFrameSRV();
        }
    }
    else
    {
        m_generator->Tick(m_context.Get());
        activeSRV = m_generator->GetShaderResourceView();
    }

    if (activeSRV)
    {
        ID3D11ShaderResourceView* safePipelineInputSRV = m_dataBus->StageAndGetSafeSRV(activeSRV);
        
        ComPtr<ID3D11DeviceContext4> context4;
        m_context.As(&context4);
        context4->Signal(m_frameCompletionFence.Get(), ++m_frameFenceValue);
        
        FrameData frameData;
        frameData.pDevice = m_device.Get();
        frameData.pContext = m_pipelineContext.Get();
        frameData.pInputSRV = safePipelineInputSRV; 
        frameData.pOutputRTV = m_finalFrameRTV.Get();
        frameData.pConsole = m_console.get();

        if (m_app_state.active_mode != ProcessingMode::Passthrough && !m_app_state.is_paused)
        {
            ComPtr<ID3D11DeviceContext4> pipelineContext4;
            m_pipelineContext.As(&pipelineContext4);
            pipelineContext4->Wait(m_frameCompletionFence.Get(), m_frameFenceValue);

            m_face_detector_module->Process(frameData);
            if (m_app_state.active_mode == ProcessingMode::Swap)
            {
                m_face_embedder_module->Process(frameData);
                m_face_swap->Process(frameData);
                
                ComPtr<ID3D11Resource> finalResource;
                m_finalFrameTexture.As(&finalResource);
                m_context->CopyResource(finalResource.Get(), m_face_swap->GetGeneratedSwappedFaceTexture());
            }
            else if (m_app_state.active_mode == ProcessingMode::BoundingBox)
            {
                m_bbox_renderer->Render(m_pipelineContext.Get(), frameData);
            }

            ComPtr<ID3D11CommandList> commandList;
            m_pipelineContext->FinishCommandList(FALSE, &commandList);
            m_context->ExecuteCommandList(commandList.Get(), FALSE);

        }
        else
        {
            m_context->CopyResource(m_finalFrameTexture.Get(), m_dataBus->GetInternalTexture());
        }
    }
    else
    {
        if (m_finalFrameRTV)
        {
            const float clearColor[] = {0.0f, 0.02f, 0.04f, 1.0f};
            m_context->ClearRenderTargetView(m_finalFrameRTV.Get(), clearColor);
        }
    }
    
    ID3D11RenderTargetView *rtv = m_swapChain->GetRenderTargetView();
    m_context->OMSetRenderTargets(1, &rtv, nullptr);
    const float clear_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    m_context->ClearRenderTargetView(rtv, clear_color);

    m_console->Render();

    ImGui::Render();
    ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
    
    if (m_producer)
    {
        m_producer->PublishFrame();
    }

    m_swapChain->Present();
}

void Application::CreateFinalFrameResources()
{
    m_finalFrameTexture.Reset();
    m_finalFrameRTV.Reset();
    m_finalFrameSRV.Reset();

    const int fixed_size = 1024;
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = fixed_size;
    desc.Height = fixed_size;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
    m_device->CreateTexture2D(&desc, nullptr, &m_finalFrameTexture);
    m_device->CreateRenderTargetView(m_finalFrameTexture.Get(), nullptr, &m_finalFrameRTV);
    m_device->CreateShaderResourceView(m_finalFrameTexture.Get(), nullptr, &m_finalFrameSRV);

    m_frame_width = fixed_size;
    m_frame_height = fixed_size;
}

void Application::SetupUITheme()
{
    ImVec4 *colors = ImGui::GetStyle().Colors;
    colors[ImGuiCol_WindowBg] = ImVec4(0.07f, 0.07f, 0.07f, 1.00f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.05f, 0.05f, 0.05f, 1.00f);
    ImVec4 orange_base = ImVec4(0.95f, 0.55f, 0.10f, 1.00f);
    ImVec4 orange_hover = ImVec4(1.00f, 0.65f, 0.20f, 1.00f);
    ImVec4 orange_active = ImVec4(1.00f, 0.50f, 0.00f, 1.00f);
    colors[ImGuiCol_Header] = orange_base;
    colors[ImGuiCol_HeaderHovered] = orange_hover;
    colors[ImGuiCol_HeaderActive] = orange_active;
    colors[ImGuiCol_TitleBgActive] = orange_base;
    colors[ImGuiCol_CheckMark] = orange_base;
    colors[ImGuiCol_SliderGrab] = orange_base;
    colors[ImGuiCol_SliderGrabActive] = orange_active;
    colors[ImGuiCol_Button] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.35f, 0.35f, 0.35f, 1.00f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.45f, 0.45f, 0.45f, 1.00f);
    colors[ImGuiCol_Separator] = ImVec4(0.40f, 0.40f, 0.40f, 1.00f);
    ImGuiStyle &style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;
    style.FrameRounding = 2.0f;
    style.GrabRounding = 2.0f;
    style.WindowBorderSize = 0.0f;
    style.FrameBorderSize = 0.0f;
    style.PopupBorderSize = 0.0f;
    style.FramePadding = ImVec2(8, 4);
    style.ItemSpacing = ImVec2(8, 4);
    style.WindowPadding = ImVec2(8, 8);
}

LRESULT CALLBACK Application::WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    Application *app = GetThisFromHandle(hWnd);
    if (msg == WM_CREATE)
    {
        LPCREATESTRUCT pCreateStruct = reinterpret_cast<LPCREATESTRUCT>(lParam);
        SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pCreateStruct->lpCreateParams));
        return 0;
    }

    if (app)
    {
        switch (msg)
        {
        case WM_TIMER:
            if (wParam == app->m_discoveryTimer)
                app->m_consumer->DiscoverStreams();
            return 0;
        case WM_SIZE:
            if (app->m_device && wParam != SIZE_MINIMIZED)
            {
                if (!app->m_swapChain->IsFullscreen())
                {
                    app->m_width = LOWORD(lParam);
                    app->m_height = HIWORD(lParam);
                    app->m_swapChain->Resize(app->m_width, app->m_height);
                }
            }
            return 0;
        case WM_SYSCOMMAND:
            if ((wParam & 0xfff0) == SC_KEYMENU)
                return 0;
            break;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        }
    }
    return DefWindowProc(hWnd, msg, wParam, lParam);
}
}