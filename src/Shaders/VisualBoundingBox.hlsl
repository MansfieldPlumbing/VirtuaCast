#include "shared.h"

StructuredBuffer<DetectedFace> g_FaceData : register(t0);
RWTexture2D<float4> g_OutputOverlay : register(u0);

#define LINE_THICKNESS 1.0
#define BBOX_COLOR float4(0.0, 1.0, 0.0, 1.0)

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

    float4 bbox = face.bbox;
    float2 pixel_coord = (float2)DTid;

    bool on_left   = (pixel_coord.x >= bbox.x - LINE_THICKNESS && pixel_coord.x <= bbox.x + LINE_THICKNESS && pixel_coord.y >= bbox.y && pixel_coord.y <= bbox.w);
    bool on_right  = (pixel_coord.x >= bbox.z - LINE_THICKNESS && pixel_coord.x <= bbox.z + LINE_THICKNESS && pixel_coord.y >= bbox.y && pixel_coord.y <= bbox.w);
    bool on_top    = (pixel_coord.y >= bbox.y - LINE_THICKNESS && pixel_coord.y <= bbox.y + LINE_THICKNESS && pixel_coord.x >= bbox.x && pixel_coord.x <= bbox.z);
    bool on_bottom = (pixel_coord.y >= bbox.w - LINE_THICKNESS && pixel_coord.y <= bbox.w + LINE_THICKNESS && pixel_coord.x >= bbox.x && pixel_coord.x <= bbox.z);

    if (on_left || on_right || on_top || on_bottom)
    {
        g_OutputOverlay[DTid] = BBOX_COLOR;
    }
}