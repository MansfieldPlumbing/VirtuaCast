// File: src/VirtuaCast/Console.h

#pragma once

#include <memory>
#include <string>
#include "imgui.h" // FIX: Include the ImGui header to define IM_FMTARGS

namespace VirtuaCast {

    class Console {
    public:
        Console();
        ~Console();

        // Adds a new line of text to the console's log buffer.
        void AddLog(const char* fmt, ...) IM_FMTARGS(2);

        // Renders the console and its slide-down handle.
        void Render();
        
        // Toggles the console's open/closed state.
        void Toggle();

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };

} // namespace VirtuaCast