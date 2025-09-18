// File: src/VirtuaCast/UI.h

#pragma once

#include "KeyCommands.h"
#include "Consumer.h"
#include "CameraSource.h"

struct ID3D11ShaderResourceView;

namespace VirtuaCast {
    
    class SourceFaceManager; // Forward declaration

    enum class ActiveSource {
        Generator,
        Camera,
        SharedStream
    };

    class UI {
    public:
        void Render(
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
        );

    private:
        void RenderRightPane(
            AppState& state,
            Consumer& consumer,
            CameraSource& camera_source,
            SourceFaceManager& source_face_manager,
            int& active_stream_index_ref,
            int& active_camera_index_ref,
            ActiveSource& active_source_ref
        );
        void RenderPreviewPane(ID3D11ShaderResourceView* final_frame_srv);
        void RenderStatusBar(const AppState& state, UINT frame_width, UINT frame_height);
    };

} // namespace VirtuaCast