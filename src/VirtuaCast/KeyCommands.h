#pragma once
#include <string>

namespace VirtuaCast {

    enum class ProcessingMode {
        Passthrough,
        Swap,
        DebugSwapArea,
        BoundingBox,
    };

    struct AppState {
        bool should_quit = false;
        bool is_paused = false;
        ProcessingMode active_mode = ProcessingMode::Passthrough;
        int source_face_change_delta = 0;
        bool trigger_run_debug_trace = false;
        bool trigger_toggle_console = false;
        bool trigger_toggle_fullscreen = false;
        int resolution_change_delta = 0;
    };

    class KeyCommandHandler {
    public:
        void ProcessInput(AppState& state);
    };

}