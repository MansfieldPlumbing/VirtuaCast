// A:/Face-On/src/Shaders/PreprocessInswapper.hlsl
//
// DOCTRINAL REFACTOR:
// - Target: Shader Model 6.2+ for f16tof32/f32tof16 intrinsics.
// - Precision: Uses native 'half' (FP16) for calculations, reducing complexity and improving performance.
// - Compiler: Written for dxc.exe.

#include "shared.h"

#define MODEL_INPUT_SIZE 128

cbuffer PreprocessCB : register(b0)
{
    TransformConstants transform;
};

// Input: The aligned full-resolution face patch
Texture2D<float4> g_InputAlignedBGRA : register(t0);
SamplerState g_LinearSampler : register(s0);

// Output: NCHW float16 buffer for the ONNX model
RWByteAddressBuffer g_ModelInputBuffer : register(u0);

[numthreads(8, 8, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= MODEL_INPUT_SIZE || DTid.y >= MODEL_INPUT_SIZE) return;

    // Use the inverse matrix from the cbuffer instance
    float2 warped_coords;
    warped_coords.x = transform.inverseTransformMatrix._11 * DTid.x + transform.inverseTransformMatrix._12 * DTid.y + transform.inverseTransformMatrix._13;
    warped_coords.y = transform.inverseTransformMatrix._21 * DTid.x + transform.inverseTransformMatrix._22 * DTid.y + transform.inverseTransformMatrix._23;

    uint width, height;
    g_InputAlignedBGRA.GetDimensions(width, height);
    float2 uv = warped_coords / float2(width, height);

    // If the warped coordinate is outside the source texture, write black (0) to the model input
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
        uint pixel_index = DTid.y * MODEL_INPUT_SIZE + DTid.x;
        uint base_address = pixel_index * 2; // 2 bytes for float16
        uint bytes_per_channel = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 2;
        
        g_ModelInputBuffer.Store(bytes_per_channel * 0 + base_address, 0);
        g_ModelInputBuffer.Store(bytes_per_channel * 1 + base_address, 0);
        g_ModelInputBuffer.Store(bytes_per_channel * 2 + base_address, 0);
        return;
    }
    
    // Sample the source texture
    float4 bgra_pixel = g_InputAlignedBGRA.SampleLevel(g_LinearSampler, uv, 0);

    // Normalize to [-1, 1] and prepare BGR channels as FP16
    half b = (half)((bgra_pixel.r * 2.0f) - 1.0f);
    half g = (half)((bgra_pixel.g * 2.0f) - 1.0f);
    half r = (half)((bgra_pixel.b * 2.0f) - 1.0f);

    // Write to NCHW buffer (float16 is 2 bytes)
    uint pixel_index = DTid.y * MODEL_INPUT_SIZE + DTid.x;
    uint base_address = pixel_index * 2;
    uint bytes_per_channel = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 2;

    // Use asuint() to get the raw 16-bit representation of the half-precision floats
    g_ModelInputBuffer.Store(bytes_per_channel * 0 + base_address, asuint(b));
    g_ModelInputBuffer.Store(bytes_per_channel * 1 + base_address, asuint(g));
    g_ModelInputBuffer.Store(bytes_per_channel * 2 + base_address, asuint(r));
}