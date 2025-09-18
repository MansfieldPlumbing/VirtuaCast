#include "KeyCommands.h"
#include "imgui.h"

namespace VirtuaCast {

    void KeyCommandHandler::ProcessInput(AppState& state) {
        ImGuiIO& io = ImGui::GetIO();
        
        state.source_face_change_delta = 0;
        if (io.MouseWheel != 0.0f) {
            state.source_face_change_delta = io.MouseWheel > 0.0f ? 1 : -1;
        }

        if (io.WantCaptureKeyboard) return;

        state.trigger_run_debug_trace = false;
        state.trigger_toggle_console = false;
        state.trigger_toggle_fullscreen = false;
        state.resolution_change_delta = 0;
        
        if (ImGui::IsKeyPressed(ImGuiKey_Escape) || ImGui::IsKeyPressed(ImGuiKey_Q)) {
            state.should_quit = true;
            return;
        }

        if (ImGui::IsKeyPressed(ImGuiKey_Space)) state.is_paused = !state.is_paused;
        if (ImGui::IsKeyPressed(ImGuiKey_GraveAccent)) state.trigger_toggle_console = true;
        if (ImGui::IsKeyPressed(ImGuiKey_Enter)) state.trigger_toggle_fullscreen = true;

        if (ImGui::IsKeyPressed(ImGuiKey_S)) state.active_mode = ProcessingMode::Swap;
        if (ImGui::IsKeyPressed(ImGuiKey_V)) state.active_mode = ProcessingMode::BoundingBox;
        if (ImGui::IsKeyPressed(ImGuiKey_P)) state.active_mode = ProcessingMode::Passthrough;
        
        if (ImGui::IsKeyPressed(ImGuiKey_D)) state.trigger_run_debug_trace = true;

        if (ImGui::IsKeyPressed(ImGuiKey_Equal) || ImGui::IsKeyPressed(ImGuiKey_KeypadAdd)) state.resolution_change_delta = 1;
        if (ImGui::IsKeyPressed(ImGuiKey_Minus) || ImGui::IsKeyPressed(ImGuiKey_KeypadSubtract)) state.resolution_change_delta = -1;
    }

}