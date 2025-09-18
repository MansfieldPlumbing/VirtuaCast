// File: src/VirtuaCast/Console.cpp

#include "Console.h"
#include "imgui.h"
#include "imgui_internal.h"
#include <vector>
#include <string>
#include <cmath>

namespace {
    float EaseOutCubic(float t) {
        return 1.0f - powf(1.0f - t, 3.0f);
    }
}

namespace VirtuaCast {

struct Console::Impl {
    ImGuiTextBuffer     Buf;
    bool                ScrollToBottom;
    bool                IsOpen;
    float               AnimationProgress; // 0.0 = closed, 1.0 = open
    bool                IsDragging;

    Impl() : ScrollToBottom(false), IsOpen(false), AnimationProgress(0.0f), IsDragging(false) {}

    void Clear() { Buf.clear(); }

    void AddLog(const char* fmt, va_list args) {
        Buf.appendfv(fmt, args);
        Buf.append("\n");
        ScrollToBottom = true;
    }

    void RenderConsoleWindow(float target_y, float target_height) {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        float window_width = viewport->Size.x;
        
        ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x, viewport->Pos.y + target_y));
        ImGui::SetNextWindowSize(ImVec2(window_width, target_height));
        
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.02f, 0.02f, 0.02f, 0.90f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);

        if (ImGui::Begin("Console", &IsOpen, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse)) {
            if (ImGui::BeginChild("ScrollingRegion", ImVec2(0, -ImGui::GetFrameHeightWithSpacing()), false, ImGuiWindowFlags_HorizontalScrollbar)) {
                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 1));
                ImGui::TextUnformatted(Buf.begin(), Buf.end());

                if (ScrollToBottom || (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()))
                    ImGui::SetScrollHereY(1.0f);
                ScrollToBottom = false;

                ImGui::PopStyleVar();
            }
            ImGui::EndChild();
        }
        ImGui::End();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(2);
    }

    void RenderHandle(float parent_y, float parent_height) {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        const float handle_height = 22.0f;
        const float handle_width = 120.0f;
        const float handle_y_pos = parent_y + parent_height;
        const float handle_x_pos = viewport->Pos.x + viewport->Size.x - handle_width - 20.0f;

        ImGui::SetNextWindowPos(ImVec2(handle_x_pos, viewport->Pos.y + handle_y_pos));
        ImGui::SetNextWindowSize(ImVec2(handle_width, handle_height));

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.15f, 0.18f, 0.90f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.95f, 0.55f, 0.10f, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1.00f, 0.50f, 0.00f, 1.00f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0,0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);

        if (ImGui::Begin("ConsoleHandle", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoBackground)) {
            ImGui::Button(IsOpen ? "== Console ==" : "++ Console ++", ImVec2(handle_width, handle_height));
            
            if (ImGui::IsItemClicked()) {
                IsOpen = !IsOpen;
                IsDragging = false;
            }

            if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                IsDragging = true;
                const float console_max_height = 500.0f;
                AnimationProgress = ImClamp(AnimationProgress + ImGui::GetIO().MouseDelta.y / console_max_height, 0.0f, 1.0f);
            } else {
                IsDragging = false;
            }
        }
        ImGui::End();
        ImGui::PopStyleVar(4);
        ImGui::PopStyleColor(3);
    }
};

Console::Console() : pImpl(std::make_unique<Impl>()) {}
Console::~Console() = default;

void Console::Toggle() {
    pImpl->IsOpen = !pImpl->IsOpen;
}

void Console::AddLog(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    pImpl->AddLog(fmt, args);
    va_end(args);
}

void Console::Render() {
    if (!pImpl->IsDragging) {
        float delta_time = ImGui::GetIO().DeltaTime;
        const float animation_speed = 6.0f;
        if (pImpl->IsOpen && pImpl->AnimationProgress < 1.0f) {
            pImpl->AnimationProgress = ImMin(1.0f, pImpl->AnimationProgress + delta_time * animation_speed);
        } else if (!pImpl->IsOpen && pImpl->AnimationProgress > 0.0f) {
            pImpl->AnimationProgress = ImMax(0.0f, pImpl->AnimationProgress - delta_time * animation_speed);
        }
    }
    
    if (pImpl->AnimationProgress > 0.0f) {
        const float console_height = 500.0f; // Increased height
        float eased_progress = EaseOutCubic(pImpl->AnimationProgress);
        float current_height = console_height * eased_progress;
        float current_y_offset = current_height - console_height;
        
        pImpl->RenderConsoleWindow(current_y_offset, console_height);
        pImpl->RenderHandle(current_y_offset, current_height);
    }
}

} // namespace VirtuaCast