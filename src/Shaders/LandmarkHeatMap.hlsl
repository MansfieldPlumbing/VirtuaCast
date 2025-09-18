#include "shared.h"

StructuredBuffer<DetectedFace> g_FaceData : register(t0);
RWTexture2D<float4> g_OutputOverlay : register(u0);

#define HEATMAP_RADIUS 30.0
#define HEATMAP_INTENSITY 0.001

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

    float total_intensity = 0.0;

    for (int i = 0; i < 5; i++)
    {
        float2 kps = face.kps[i];
        float dist = distance(float2(DTid), kps);
        
        // Use a Gaussian-like falloff for intensity
        float intensity = exp(-dist * dist / (HEATMAP_RADIUS * HEATMAP_RADIUS));
        total_intensity += intensity;
    }

    if (total_intensity > 0.01)
    {
        // Atomically add to the red channel to build up the heatmap
        // Note: This requires a render target that supports atomic operations.
        // For simplicity here, we do a direct write. A true atomic implementation
        // would use an RWTexture2D<uint> and InterlockedAdd.
        // This visualizes the intensity directly.
        float4 current_color = g_OutputOverlay[DTid];
        float blended_intensity = saturate(current_color.r + total_intensity * HEATMAP_INTENSITY);
        g_OutputOverlay[DTid] = float4(blended_intensity, 0, 0, blended_intensity);
    }
}