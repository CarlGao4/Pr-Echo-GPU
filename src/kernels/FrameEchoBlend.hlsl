// Copyright (c) 2026 CarlGao4
// Licensed under the MIT License. See LICENSE for details.
// github.com/CarlGao4/Pr-Echo-GPU

//=============================================================================
// FrameEchoBlend.hlsl - DirectX Compute Shader
//
// Architecture:
//   1. Entry kernel reads raw pixel data from buffer (any pixel format)
//   2. Converts to float4 RGBA (standardized blend space)
//   3. Calls blend function for each temporal sample
//   4. Converts result back to original pixel format
//   5. Writes to output buffer
//
// The five BlendNNNN() functions operate on float4 RGBA only -
// they have no knowledge of pixel formats.
//=============================================================================

// ============================================================================
// Pixel format bitfield (uint32)
//   Bits 0-4 (0x1F): pixel arrangement
//   Bits 5-7 (0xE0): data type
//   Bits 8-31:       reserved
// ============================================================================
#define PIXEL_ARR_VUYA  0   // VUYA 4:4:4 (Premiere's PrPixelFormat_VUYA_4444)
#define PIXEL_ARR_BGRA  1   // BGRA interleaved (common Windows format)
#define PIXEL_ARR_ARGB  2   // ARGB interleaved (AE's PF_Pixel_ARGB)
#define PIXEL_ARR_MASK  0x1F

#define PIXEL_TYPE_U8    (0 << 5)  // 8-bit  normalized  [0,255] -> [0,1]
#define PIXEL_TYPE_U16   (1 << 5)  // 16-bit normalized  [0,32768] -> [0,1]
#define PIXEL_TYPE_F32   (2 << 5)  // 32-bit float (may exceed [0,1] for HDR)
#define PIXEL_TYPE_F16   (3 << 5)  // 16-bit half-float (fp16), common for GPU 16f
#define PIXEL_TYPE_MASK  0xE0

// Blend mode constants
#define BLEND_MODE_NEW_ON_TOP    0
#define BLEND_MODE_NEW_ON_BOTTOM 1
#define BLEND_MODE_ADD           2
#define BLEND_MODE_MAXIMUM       3
#define BLEND_MODE_MINIMUM       4

// ============================================================================
// Premultiply / Unpremultiply helpers (RGBA space)
// ============================================================================

float4 Premultiply(float4 pixel, float opacity)
{
    float alpha = saturate(pixel.a * opacity);
    return float4(pixel.rgb * alpha, alpha);
}

float4 Unpremultiply(float4 premul)
{
    if (premul.a <= 0.0) return 0.0;
    float clamped = saturate(premul.a);
    float inv = rcp(clamped);
    return float4(premul.rgb * inv, clamped);
}

// ============================================================================
// Five blend-mode kernels - operate on float4 RGBA only
// ============================================================================

float4 BlendNewOnTop(float4 acc, float4 premul)
{
    return mad(acc, 1.0 - premul.a, premul);
}

float4 BlendNewOnBottom(float4 acc, float4 premul)
{
    return mad(premul, 1.0 - acc.a, acc);
}

float4 BlendAdd(float4 acc, float4 premul)
{
    return acc + premul;
}

float4 BlendMaximum(float4 acc, float4 premul)
{
    return max(acc, premul);
}

float4 BlendMinimum(float4 acc, float4 premul)
{
    return min(acc, premul);
}

// ============================================================================
// Pixel format conversion helpers
// ============================================================================

// Byte size of one pixel in the given format
uint PixelSize(uint pf)
{
    uint type = pf & PIXEL_TYPE_MASK;
    if (type == PIXEL_TYPE_U8)  return 4;
    if (type == PIXEL_TYPE_U16) return 8;
    if (type == PIXEL_TYPE_F16) return 8;
    return 16; // PIXEL_TYPE_F32
}

// Read raw bytes from buffer, return float4 in the storage-native color space
// (VUYA: V,U,Y,A - BGRA: B,G,R,A - ARGB: A,R,G,B)
float4 ReadRawPixel(ByteAddressBuffer buf, uint byteOffset, uint pf)
{
    uint type = pf & PIXEL_TYPE_MASK;

    if (type == PIXEL_TYPE_F32)
    {
        return buf.Load<float4>(byteOffset);
    }
    else if (type == PIXEL_TYPE_F16)
    {
        // half4 stored as 2 x uint32 (4 x f16) -> convert to float4
        uint lo = buf.Load(byteOffset);
        uint hi = buf.Load(byteOffset + 4);
        float4 result;
        result.x = f16tof32(lo & 0xFFFF);
        result.y = f16tof32((lo >> 16) & 0xFFFF);
        result.z = f16tof32(hi & 0xFFFF);
        result.w = f16tof32((hi >> 16) & 0xFFFF);
        return result;
    }
    else if (type == PIXEL_TYPE_U16)
    {
        // 4 * uint16 little-endian, [0..32768] -> [0..1]
        uint lo = buf.Load(byteOffset);
        uint hi = buf.Load(byteOffset + 4);
        return float4(
            (lo & 0xFFFF) * (1.0 / 32768.0),
            ((lo >> 16) & 0xFFFF) * (1.0 / 32768.0),
            (hi & 0xFFFF) * (1.0 / 32768.0),
            ((hi >> 16) & 0xFFFF) * (1.0 / 32768.0)
        );
    }
    else // PIXEL_TYPE_U8
    {
        // 4 * uint8, [0..255] -> [0..1]
        uint raw = buf.Load(byteOffset);
        return float4(
            (raw & 0xFF) * (1.0 / 255.0),
            ((raw >> 8) & 0xFF) * (1.0 / 255.0),
            ((raw >> 16) & 0xFF) * (1.0 / 255.0),
            ((raw >> 24) & 0xFF) * (1.0 / 255.0)
        );
    }
}

// Convert from storage-native color space to standard float4 RGBA (R=x,G=y,B=z,A=w)
float4 StorageToRGBA(float4 pixel, uint pf)
{
    uint arr = pf & PIXEL_ARR_MASK;

    if (arr == PIXEL_ARR_BGRA)
    {
        // BGRA: x=B, y=G, z=R, w=A -> RGBA: x=R, y=G, z=B, w=A
        return pixel.zyxw;
    }
    else if (arr == PIXEL_ARR_ARGB)
    {
        // ARGB: x=A, y=R, z=G, w=B -> RGBA: x=R, y=G, z=B, w=A
        return pixel.yzwx;
    }
    else // PIXEL_ARR_VUYA
    {
        // VUYA: x=V, y=U, z=Y, w=A -> YUV->RGB
        float v = pixel.x, u = pixel.y, y = pixel.z, a = pixel.w;
        return float4(
            y + 1.402f * v,                         // R
            y - 0.344136f * u - 0.714136f * v,      // G
            y + 1.772f * u,                         // B
            a
        );
    }
}

// Convert from float4 RGBA back to storage-native color space
float4 RGBAToStorage(float4 rgba, uint pf)
{
    uint arr = pf & PIXEL_ARR_MASK;

    if (arr == PIXEL_ARR_BGRA)
    {
        return rgba.zyxw;
    }
    else if (arr == PIXEL_ARR_ARGB)
    {
        return rgba.wxyz;
    }
    else // PIXEL_ARR_VUYA
    {
        float r = rgba.x, g = rgba.y, b = rgba.z, a = rgba.w;
        return float4(
            0.5f * r - 0.418688f * g - 0.081312f * b,   // V
            -0.168736f * r - 0.331264f * g + 0.5f * b,  // U
            0.299f * r + 0.587f * g + 0.114f * b,       // Y
            a
        );
    }
}

// Write float4 (in storage-native space) back to buffer, quantizing as needed
void WriteRawPixel(RWByteAddressBuffer buf, uint byteOffset, float4 pixel, uint pf)
{
    uint type = pf & PIXEL_TYPE_MASK;

    if (type == PIXEL_TYPE_F32)
    {
        // Alpha must stay in [0,1]; RGB may exceed range for HDR
        pixel.a = saturate(pixel.a);
        buf.Store<float4>(byteOffset, pixel);
    }
    else if (type == PIXEL_TYPE_F16)
    {
        // float4 -> half-float using f32tof16
        pixel.a = saturate(pixel.a);
        uint2 packed;
        packed.x = f32tof16(pixel.x) | (f32tof16(pixel.y) << 16);
        packed.y = f32tof16(pixel.z) | (f32tof16(pixel.w) << 16);
        buf.Store<uint2>(byteOffset, packed);
    }
    else if (type == PIXEL_TYPE_U16)
    {
        uint lo = ((uint)(saturate(pixel.x) * 32767.5) & 0xFFFF) |
                  (((uint)(saturate(pixel.y) * 32767.5) & 0xFFFF) << 16);
        uint hi = ((uint)(saturate(pixel.z) * 32767.5) & 0xFFFF) |
                  (((uint)(saturate(pixel.w) * 32767.5) & 0xFFFF) << 16);
        buf.Store(byteOffset, lo);
        buf.Store(byteOffset + 4, hi);
    }
    else // PIXEL_TYPE_U8
    {
        uint raw = ((uint)(saturate(pixel.x) * 255.0) & 0xFF) |
                   (((uint)(saturate(pixel.y) * 255.0) & 0xFF) << 8) |
                   (((uint)(saturate(pixel.z) * 255.0) & 0xFF) << 16) |
                   (((uint)(saturate(pixel.w) * 255.0) & 0xFF) << 24);
        buf.Store(byteOffset, raw);
    }
}

// Convenience: read + convert to float4 RGBA in one call
float4 LoadPixelRGBA(ByteAddressBuffer buf, uint byteOffset, uint pf)
{
    return StorageToRGBA(ReadRawPixel(buf, byteOffset, pf), pf);
}

// Convenience: convert RGBA + write in one call
void StorePixelRGBA(RWByteAddressBuffer buf, uint byteOffset, float4 rgba, uint pf)
{
    WriteRawPixel(buf, byteOffset, RGBAToStorage(rgba, pf), pf);
}

// ============================================================================
// BlendSamples - process all samples for one pixel
// ============================================================================
float4 BlendSamples(ByteAddressBuffer samples, uint pixelRowByteOffset,
                    ByteAddressBuffer opacities, uint pf, uint pixelSize,
                    uint numSamples, int blendMode, uint sampleImageSize)
{
    float4 acc = 0.0;

    [loop]
    for (uint i = 0; i < numSamples; ++i)
    {
        uint srcOff = i * sampleImageSize + pixelRowByteOffset;
        float4 sample = LoadPixelRGBA(samples, srcOff, pf);
        float opacity = opacities.Load<float>(i * 4);
        float4 premul = Premultiply(sample, opacity);

        [branch]
        switch (blendMode)
        {
        case BLEND_MODE_NEW_ON_TOP:
            acc = BlendNewOnTop(acc, premul);
            break;
        case BLEND_MODE_NEW_ON_BOTTOM:
            acc = BlendNewOnBottom(acc, premul);
            break;
        case BLEND_MODE_ADD:
            acc = BlendAdd(acc, premul);
            break;
        case BLEND_MODE_MAXIMUM:
            acc = BlendMaximum(acc, premul);
            break;
        case BLEND_MODE_MINIMUM:
            acc = BlendMinimum(acc, premul);
            break;
        }
    }

    return Unpremultiply(acc);
}

// ============================================================================
// Resources & constant buffer
// ============================================================================
RWByteAddressBuffer mOutput     : register(u0);
ByteAddressBuffer   mSampleSets : register(t0);
ByteAddressBuffer   mOpacities  : register(t1);

cbuffer cb : register(b0)
{
    int mBoxLeft;
    int mBoxTop;
    int mBoxWidth;
    int mBoxHeight;
    int mBytePitch;         // row stride in bytes (matches host)
    int mHeight;            // full image height (for per-sample image offset calc)
    int mPixelFormat;       // bitfield: arrangement | type
    int mNumSamples;
    int mBlendMode;
};

// ============================================================================
// Entry point
// Dispatch grid = ceil(mBoxWidth/16) * ceil(mBoxHeight/16)
// SV_DispatchThreadID maps directly into the box region
// ============================================================================
[numthreads(16, 16, 1)]
[RootSignature(
    "DescriptorTable(CBV(b0), visibility=SHADER_VISIBILITY_ALL),"
    "DescriptorTable(UAV(u0), visibility=SHADER_VISIBILITY_ALL),"
    "DescriptorTable(SRV(t0), visibility=SHADER_VISIBILITY_ALL),"
    "DescriptorTable(SRV(t1), visibility=SHADER_VISIBILITY_ALL)"
)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= mBoxWidth || dtid.y >= mBoxHeight) return;

    uint2 px = uint2(mBoxLeft + dtid.x, mBoxTop + dtid.y);
    uint ps = PixelSize(mPixelFormat);
    uint sampleImageSize = mBytePitch * mHeight;
    uint rowByteOff = px.y * mBytePitch + px.x * ps;

    float4 result = BlendSamples(mSampleSets, rowByteOff,
                                 mOpacities, mPixelFormat, ps,
                                 mNumSamples, mBlendMode, sampleImageSize);

    StorePixelRGBA(mOutput, rowByteOff, result, mPixelFormat);
}
