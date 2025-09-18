// A:/Face-On/src/Shaders/CameraSetup.hlsl
//
// DOCTRINE: "The Gracious Ambassador"
// This shader acts as the diplomatic envoy between the camera and the pipeline.
// It takes the raw camera texture, which may have any resolution or aspect
// ratio (e.g., portrait 720x1280), and renders it to a standardized,
// landscape-oriented texture that the rest of the pipeline expects.
// This is a "stretch blit" operation.

#include "shared.h"

Texture2D<float4> g_InputCameraBGRA : register(t0);
RWTexture2D<float4> g_OutputFormattedBGRA : register(u0);
SamplerState g_LinearSampler : register(s0);

[numthreads(8, 8, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    uint width, height;
    g_OutputFormattedBGRA.GetDimensions(width, height);
    if (DTid.x >= width || DTid.y >= height)
    {
        return;
    }

    // Calculate the normalized UV coordinates for the output pixel.
    float2 uv = (float2(DTid.x, DTid.y) + 0.5f) / float2(width, height);

    // Sample the input camera texture at that UV coordinate.
    // The linear sampler handles the stretching automatically.
    g_OutputFormattedBGRA[DTid] = g_InputCameraBGRA.SampleLevel(g_LinearSampler, uv, 0);
}