// File: src/VirtuaCast/UI.cpp

#include "UI.h"
#include "imgui.h"
#include <string>
#include <algorithm>
#include <cwctype>
#include "SourceFace.h" // Include the new header

namespace {
    std::string WstringToUtf8(const std::wstring& wstr) {
        if (wstr.empty()) return std::string();
        int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), NULL, 0, NULL, NULL);
        std::string strTo(size_needed, 0);
        WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
        return strTo;
    }
}

namespace VirtuaCast {

void UI::Render(
    AppState& state,
    Consumer& consumer,
    CameraSource& camera_source,
    SourceFaceManager& source_face_manager,
    int& active_stream_index_ref,
    int& active_camera_index_ref,
    ActiveSource& active_source_ref,
    ID3D11ShaderResourceView* final_frame_srv,
    UINT frame_width,
    UINT frame_height
) {
    RenderPreviewPane(final_frame_srv);
    RenderRightPane(state, consumer, camera_source, source_face_manager, active_stream_index_ref, active_camera_index_ref, active_source_ref);
    RenderStatusBar(state, frame_width, frame_height);
}

void UI::RenderRightPane(
    AppState& state,
    Consumer& consumer,
    CameraSource& camera_source,
    SourceFaceManager& source_face_manager,
    int& active_stream_index_ref,
    int& active_camera_index_ref,
    ActiveSource& active_source_ref
) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float right_pane_width = 350.0f;
    const ImGuiWindowFlags pane_flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar;

    ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x + viewport->Size.x - right_pane_width, viewport->Pos.y));
    ImGui::SetNextWindowSize(ImVec2(right_pane_width, viewport->Size.y - 40.0f));
    ImGui::Begin("Controls", nullptr, pane_flags);
    
    ImGui::Text("Video Source");
    ImGui::Separator();

    if (ImGui::RadioButton("Internal Generator", active_source_ref == ActiveSource::Generator)) {
        active_source_ref = ActiveSource::Generator;
    }

    const auto& cameras = camera_source.GetAvailableCameras();
    for(size_t i = 0; i < cameras.size(); ++i) {
        std::string label = WstringToUtf8(cameras[i].name);
        if (ImGui::RadioButton(label.c_str(), active_source_ref == ActiveSource::Camera && active_camera_index_ref == (int)i)) {
            active_source_ref = ActiveSource::Camera;
            active_camera_index_ref = (int)i;
        }
    }

    ImGui::Separator();
    ImGui::Text("Shared Streams (Producers)");

    const auto& streams = consumer.GetDiscoveredStreams();
    if (streams.empty()) {
        ImGui::TextDisabled("No active producers found.");
    } else {
        for (size_t i = 0; i < streams.size(); ++i) {
            std::string label = WstringToUtf8(streams[i].processName) + " (PID: " + std::to_string(streams[i].processId) + ")";
            if (ImGui::RadioButton(label.c_str(), active_source_ref == ActiveSource::SharedStream && active_stream_index_ref == (int)i)) {
                active_source_ref = ActiveSource::SharedStream;
                active_stream_index_ref = (int)i;
            }
        }
    }
    
    ImGui::Separator();
    ImGui::Text("Pipeline Stages");

    const SourceFace* current_source = source_face_manager.GetCurrentSource();
    std::string source_name = current_source ? current_source->display_name : "None (Scroll in preview)";
    ImGui::Text("Source: %s", source_name.c_str());
    ImGui::Separator();


    if (ImGui::RadioButton("Passthrough", state.active_mode == ProcessingMode::Passthrough)) state.active_mode = ProcessingMode::Passthrough;
    if (ImGui::RadioButton("Face Detection", state.active_mode == ProcessingMode::BoundingBox)) state.active_mode = ProcessingMode::BoundingBox;
    if (ImGui::RadioButton("Face Augmentation", state.active_mode == ProcessingMode::Swap)) state.active_mode = ProcessingMode::Swap;


    ImGui::End();
}

void UI::RenderPreviewPane(ID3D11ShaderResourceView* final_frame_srv) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float right_pane_width = 350.0f;
    const ImGuiWindowFlags pane_flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar;

    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(ImVec2(viewport->Size.x - right_pane_width, viewport->Size.y));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0,0)); // Make image fill the pane
    ImGui::Begin("Preview", nullptr, pane_flags | ImGuiWindowFlags_NoBringToFrontOnFocus);
    
    if (final_frame_srv) {
        ImGui::Image((void*)final_frame_srv, ImGui::GetContentRegionAvail());
    } else {
        const char* no_source_text = "No active video source.";
        ImVec2 text_size = ImGui::CalcTextSize(no_source_text);
        ImVec2 window_size = ImGui::GetWindowSize();
        ImGui::SetCursorPos(ImVec2((window_size.x - text_size.x) * 0.5f, (window_size.y - text_size.y) * 0.5f));
        ImGui::TextUnformatted(no_source_text);
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

void UI::RenderStatusBar(const AppState& state, UINT frame_width, UINT frame_height) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float right_pane_width = 350.0f;
    const ImGuiWindowFlags pane_flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar;

    ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x + viewport->Size.x - right_pane_width, viewport->Pos.y + viewport->Size.y - 40.0f));
    ImGui::SetNextWindowSize(ImVec2(right_pane_width, 40.0f));
    ImGui::Begin("Status", nullptr, pane_flags);

    std::string mode_str = "Mode: PASSTHROUGH (P)";
    if (state.active_mode == ProcessingMode::Swap) mode_str = "Mode: SWAP (S)";
    else if (state.active_mode == ProcessingMode::BoundingBox) mode_str = "Mode: BBOX (V)";
    else if (state.active_mode == ProcessingMode::DebugSwapArea) mode_str = "Mode: DEBUG SWAP (D)";
    
    ImGui::Text("%s", mode_str.c_str());
    ImGui::SameLine();
    ImGui::Text(state.is_paused ? "| PAUSED (Space)" : "| RUNNING (Space)");
    
    if (frame_width > 0 && frame_height > 0) {
        ImGui::SameLine();
        std::string res_str = "| " + std::to_string(frame_width) + "x" + std::to_string(frame_height);
        ImGui::Text("%s", res_str.c_str());
    }
    
    ImGui::End();
}

}