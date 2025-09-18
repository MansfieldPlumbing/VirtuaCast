#include "shared.h"

StructuredBuffer<DetectedFace> g_FaceData : register(t0);
RWTexture2D<float4> g_OutputOverlay : register(u0);

#define MARKER_SIZE 3
#define LANDMARK_COLOR float4(1.0, 0.8, 0.0, 1.0) // Yellow-Orange

[numthreads(8, 8, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    uint width, height;
    g_OutputOverlay.GetDimensions(width, height);
    if (DTid.x >= width || DTid.y >= height)
    {
        return;
    }

    DetectedFace face = g_FaceData[0];
    if (face.det_score < 0.5) return;

    for (int i = 0; i < 5; i++)
    {
        float2 kps = face.kps[i];
        
        // Check if the current pixel is on the horizontal or vertical line of a cross marker
        bool on_horiz = (abs(DTid.x - kps.x) <= MARKER_SIZE && abs(DTid.y - kps.y) < 1.0);
        bool on_vert  = (abs(DTid.y - kps.y) <= MARKER_SIZE && abs(DTid.x - kps.x) < 1.0);

        if (on_horiz || on_vert)
        {
            g_OutputOverlay[DTid] = LANDMARK_COLOR;
            return; // Exit after drawing one part of a marker
        }
    }
}