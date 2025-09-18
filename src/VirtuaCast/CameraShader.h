// File: src/VirtuaCast/CameraShader.h

#pragma once

#include <vector>
#include <d3d12.h>
#include <d3dcommon.h> // Correctly defines ID3DBlob
#include <wrl/client.h>

namespace VirtuaCast {

    HRESULT GetLetterboxShader(
        Microsoft::WRL::ComPtr<ID3DBlob>& outVertexShader,
        Microsoft::WRL::ComPtr<ID3DBlob>& outPixelShader
    );

} // namespace VirtuaCast