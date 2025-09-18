Texture2D<float4> YUY2Texture : register(t0);
SamplerState      PointSampler : register(s0);

// Corrected: ps_5_0 does not allow 'half' type for global constants. Changed to 'float'.
static const float3x3 YUV_TO_RGB = {
    1.164,  1.164, 1.164,
    0.0,   -0.392, 2.017,
    1.596, -0.813, 0.0
};

float4 main(float4 pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_TARGET
{
    float4 yuyv = YUY2Texture.Sample(PointSampler, uv);
    
    // In YUY2 format, this pixel's luma (Y) is in the .g component.
    float y = yuyv.g;
    // Chroma values (U and V) are shared between pixels.
    float2 chroma = float2(yuyv.r, yuyv.b); // U is in .r, V is in .b

    float3 yuv = float3(y, chroma.x, chroma.y);
    yuv -= float3(0.0625, 0.5, 0.5); // Offset for YCbCr

    float3 rgb = mul(yuv, YUV_TO_RGB);

    // Return as BGRA for compatibility with swap chain formats.
    return float4(rgb.b, rgb.g, rgb.r, 1.0);
}