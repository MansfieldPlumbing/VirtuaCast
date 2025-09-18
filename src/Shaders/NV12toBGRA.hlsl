Texture2D<float>  LumaTexture   : register(t0);
Texture2D<float2> ChromaTexture : register(t1);
SamplerState      PointSampler  : register(s0);

// Corrected: ps_5_0 does not allow 'half' type for global constants. Changed to 'float'.
static const float3x3 YUV_TO_RGB = {
    1.164,  1.164, 1.164,
    0.0,   -0.392, 2.017,
    1.596, -0.813, 0.0
};

float4 main(float4 pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_TARGET
{
    float y = LumaTexture.Sample(PointSampler, uv).r;
    float2 uv_chroma = ChromaTexture.Sample(PointSampler, uv).rg;

    float3 yuv = float3(y, uv_chroma.x, uv_chroma.y);
    yuv -= float3(0.0625, 0.5, 0.5); // Offset for YCbCr
    
    float3 rgb = mul(yuv, YUV_TO_RGB);

    // Return as BGRA for compatibility with swap chain formats.
    return float4(rgb.b, rgb.g, rgb.r, 1.0);
}