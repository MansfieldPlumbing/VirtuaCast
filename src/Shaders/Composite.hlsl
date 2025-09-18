Texture2D<float4> TextureA : register(t0);
Texture2D<float4> TextureB : register(t1);
SamplerState Sampler : register(s0);

struct VS_OUTPUT {
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD;
};

// Replaces the non-standard 'select' intrinsic with the standard 'lerp' function.
// lerp(x, y, s) is equivalent to x * (1 - s) + y * s.
float4 main(VS_OUTPUT input) : SV_TARGET {
    float4 colorA = TextureA.Sample(Sampler, input.uv);
    float4 colorB = TextureB.Sample(Sampler, input.uv);
    
    // Use the alpha channel of the second texture as a blend mask.
    float mask = colorB.a;
    
    // Blend between colorA and colorB using the mask.
    return lerp(colorA, colorB, mask);
}