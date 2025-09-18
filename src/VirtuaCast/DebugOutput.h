#pragma once

#include "Types.h"
#include <windows.h>

namespace VirtuaCast { class Application; }

struct TegritySourceFace {
    const char* name;
    const float* normed_embedding;
};

HRESULT TegrityDebug_RunFullSwapTrace(void* app_handle, const TegritySourceFace* source_face);