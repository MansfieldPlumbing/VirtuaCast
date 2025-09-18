// A:/Face-On/src/Shaders/PostprocessBlend.hlsl
//
// DOCTRINAL REFACTOR:
// - Target: Shader Model 6.2+ compatibility.
// - Precision: Uses native 'half' (FP16) for color math.
// - FIX: Reinstated the manual f16-to-f32 conversion helper. This is the
//   robust and correct method for unpacking a uint loaded from a ByteAddressBuffer.

#include "shared.h"

#define MODEL_OUTPUT_SIZE 128

cbuffer PostprocessCB : register(b0)
{
    TransformConstants transform;
};

// Inputs
ByteAddressBuffer g_InswapperOutputBuffer : register(t0); // NCHW float16
Texture2D<float4> g_OriginalTargetFrame : register(t1);
Texture2D<float> g_BlendingMask : register(t2);
SamplerState g_LinearSampler : register(s0);

// Output
RWTexture2D<float4> g_FinalOutputImage : register(u0);

// SM 6.0+ COMPATIBILITY: Robust helper to unpack a uint holding f16 data to a float32.
float f16tof32(uint val)
{
    float result;
    uint sign = (val & 0x8000) << 16;
    uint exponent = (val >> 10) & 0x1F;
    uint mantissa = val & 0x03FF;

    if (exponent == 0x1F) { // Inf or NaN
        result = asfloat(sign | 0x7F800000 | (mantissa << 13));
    }
    else if (exponent == 0) { // Denormal or Zero
        if (mantissa == 0) {
            result = asfloat(sign); // Zero
        } else { // Denormal
            exponent = 1;
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x03FF;
            exponent += 112;
            result = asfloat(sign | (exponent << 23) | (mantissa << 13));
        }
    }
    else { // Normal
        exponent += 112;
        result = asfloat(sign | (exponent << 23) | (mantissa << 13));
    }
    return result;
}

[numthreads(8, 8, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    uint width, height;
    g_FinalOutputImage.GetDimensions(width, height);
    if (DTid.x >= width || DTid.y >= height) return;

    float2 full_res_uv = (float2)(DTid.xy + 0.5f) / float2(width, height);
    half4 original_pixel = (half4)g_OriginalTargetFrame.SampleLevel(g_LinearSampler, full_res_uv, 0);

    // Calculate the coordinate to sample from the 128x128 swapped face patch
    float2 swapped_face_coords;
    swapped_face_coords.x = transform.inverseTransformMatrix._11 * DTid.x + transform.inverseTransformMatrix._12 * DTid.y + transform.inverseTransformMatrix._13;
    swapped_face_coords.y = transform.inverseTransformMatrix._21 * DTid.x + transform.inverseTransformMatrix._22 * DTid.y + transform.inverseTransformMatrix._23;

    half4 swapped_pixel = half4(original_pixel.rgb, 0.0h);
    if (swapped_face_coords.x >= 0.0f && swapped_face_coords.x < MODEL_OUTPUT_SIZE &&
        swapped_face_coords.y >= 0.0f && swapped_face_coords.y < MODEL_OUTPUT_SIZE)
    {
        uint2 swapped_int_coords = (uint2)swapped_face_coords;
        uint pixel_index = swapped_int_coords.y * MODEL_OUTPUT_SIZE + swapped_int_coords.x;
        uint base_address = pixel_index * 2; // Address is in bytes
        uint bytes_per_channel = MODEL_OUTPUT_SIZE * MODEL_OUTPUT_SIZE * 2;
        
        // Load the 32-bit uint values that contain our 16-bit float data
        uint b16_uint = g_InswapperOutputBuffer.Load(bytes_per_channel * 0 + base_address);
        uint g16_uint = g_InswapperOutputBuffer.Load(bytes_per_channel * 1 + base_address);
        uint r16_uint = g_InswapperOutputBuffer.Load(bytes_per_channel * 2 + base_address);

        // Manually convert the uints to floats, then cast to half
        half b = (half)((f16tof32(b16_uint) + 1.0f) * 0.5f);
        half g = (half)((f16tof32(g16_uint) + 1.0f) * 0.5f);
        half r = (half)((f16tof32(r16_uint) + 1.0f) * 0.5f);
        
        swapped_pixel = half4(b, g, r, 1.0h);
    }
    
    // Blend using the mask
    half mask_alpha = (half)g_BlendingMask.SampleLevel(g_LinearSampler, full_res_uv, 0).r;
    half3 blended_rgb = lerp(original_pixel.rgb, swapped_pixel.rgb, mask_alpha);
    
    g_FinalOutputImage[DTid] = float4(blended_rgb, original_pixel.a);
}