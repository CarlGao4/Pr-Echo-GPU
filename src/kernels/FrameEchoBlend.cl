// Copyright (c) 2026 CarlGao4
// Licensed under the MIT License. See LICENSE for details.
// github.com/CarlGao4/Pr-Echo-GPU

//=============================================================================
// FrameEchoBlend.cl - OpenCL Kernel
//
// Architecture:
//   1. Entry kernel reads raw pixel data from buffer (any pixel format)
//   2. Converts to float4 RGBA (standardized blend space)
//   3. Calls blend function for each temporal sample
//   4. Converts result back to original pixel format
//   5. Writes to output buffer
//
// The five BlendNNNN_Kernel() functions operate on float4 RGBA only.
//
// Compilation note:
//   This file is designed to be preprocessed by MSVC (/TP /EP) and then
//   embedded as a C string via CreateCString.py.  The GF_KERNEL_FUNCTION
//   macro abstracts the OpenCL/CUDA kernel declaration differences.
//   Define GF_DEVICE_TARGET_OPENCL=1 or GF_DEVICE_TARGET_CUDA=1 before
//   including PrGPU/KernelSupport/KernelCore.h.
//=============================================================================

#ifndef FRAME_ECHO_BLEND_CL
#define FRAME_ECHO_BLEND_CL

#include "PrGPU/KernelSupport/KernelCore.h"
#include "PrGPU/KernelSupport/KernelMemory.h"

// ============================================================================
// Pixel format bitfield (uint32)
//   Bits 0-4 (0x1F): pixel arrangement
//   Bits 5-7 (0xE0): data type
//   Bits 8-31:       reserved
// ============================================================================
#define PIXEL_ARR_VUYA  0
#define PIXEL_ARR_BGRA  1
#define PIXEL_ARR_ARGB  2
#define PIXEL_ARR_MASK  0x1F

#define PIXEL_TYPE_U8    (0 << 5)
#define PIXEL_TYPE_U16   (1 << 5)
#define PIXEL_TYPE_F32   (2 << 5)
#define PIXEL_TYPE_F16   (3 << 5)
#define PIXEL_TYPE_MASK  0xE0

// Blend mode constants
#define BLEND_NEW_ON_TOP    0
#define BLEND_NEW_ON_BOTTOM 1
#define BLEND_ADD           2
#define BLEND_MAXIMUM       3
#define BLEND_MINIMUM       4

// ============================================================================
// Premultiply / Unpremultiply helpers (float4 RGBA)
// ============================================================================

static inline float Clamp01(float v)
{
    return fmax(0.0f, fmin(1.0f, v));
}

static inline float4 PremultiplyKernel(float4 pixel, float opacity)
{
    float alpha = Clamp01(pixel.w * opacity);
    return (float4)(pixel.x * alpha, pixel.y * alpha, pixel.z * alpha, alpha);
}

static inline float4 UnpremultiplyKernel(float4 premul)
{
    if (premul.w <= 0.0f) return (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float clamped = Clamp01(premul.w);
    float inv = 1.0f / clamped;
    return (float4)(premul.x * inv, premul.y * inv, premul.z * inv, clamped);
}

// ============================================================================
// Five blend functions - operate on float4 RGBA only
// ============================================================================

static inline float4 BlendNewOnTopKernel(float4 acc, float4 premul)
{
    float omsa = 1.0f - premul.w;
    return (float4)(
        premul.x + acc.x * omsa,
        premul.y + acc.y * omsa,
        premul.z + acc.z * omsa,
        premul.w + acc.w * omsa
    );
}

static inline float4 BlendNewOnBottomKernel(float4 acc, float4 premul)
{
    float omsa = 1.0f - acc.w;
    return (float4)(
        acc.x + premul.x * omsa,
        acc.y + premul.y * omsa,
        acc.z + premul.z * omsa,
        acc.w + premul.w * omsa
    );
}

static inline float4 BlendAddKernel(float4 acc, float4 premul)
{
    return acc + premul;
}

static inline float4 BlendMaximumKernel(float4 acc, float4 premul)
{
    return fmax(acc, premul);
}

static inline float4 BlendMinimumKernel(float4 acc, float4 premul)
{
    return fmin(acc, premul);
}

// ============================================================================
// Pixel format conversion helpers
// ============================================================================

static inline int PixelSizeOCL(int pf)
{
    int type = pf & PIXEL_TYPE_MASK;
    if (type == PIXEL_TYPE_U8)  return 4;
    if (type == PIXEL_TYPE_U16) return 8;
    if (type == PIXEL_TYPE_F16) return 8;
    return 16;
}

// Read raw pixel (byte buffer) -> float4 in storage-native color space
static inline float4 ReadRawPixelOCL(__global const uchar* base, uint byteOff, int pf)
{
    int type = pf & PIXEL_TYPE_MASK;

    if (type == PIXEL_TYPE_F32)
    {
        __global const float* p = (__global const float*)(base + byteOff);
        return vload4(0, p);
    }
    else if (type == PIXEL_TYPE_F16)
    {
        // half4 -> float4
        __global const half* p = (__global const half*)(base + byteOff);
        return (float4)((float)p[0], (float)p[1], (float)p[2], (float)p[3]);
    }
    else if (type == PIXEL_TYPE_U16)
    {
        __global const ushort* p = (__global const ushort*)(base + byteOff);
        return (float4)(
            p[0] * (1.0f / 32768.0f),
            p[1] * (1.0f / 32768.0f),
            p[2] * (1.0f / 32768.0f),
            p[3] * (1.0f / 32768.0f)
        );
    }
    else // PIXEL_TYPE_U8
    {
        uchar4 raw = vload4(0, base + byteOff);
        return convert_float4(raw) * (1.0f / 255.0f);
    }
}

// Storage-native float4 -> standard float4 RGBA (R=x, G=y, B=z, A=w)
static inline float4 StorageToRGBAOCL(float4 pixel, int pf)
{
    int arr = pf & PIXEL_ARR_MASK;

    if (arr == PIXEL_ARR_BGRA)
    {
        return pixel.zyxw;
    }
    else if (arr == PIXEL_ARR_ARGB)
    {
        return pixel.yzwx;
    }
    else // PIXEL_ARR_VUYA
    {
        float v = pixel.x, u = pixel.y, y = pixel.z, a = pixel.w;
        return (float4)(
            y + 1.402f * v,
            y - 0.344136f * u - 0.714136f * v,
            y + 1.772f * u,
            a
        );
    }
}

// float4 RGBA -> storage-native float4
static inline float4 RGBAToStorageOCL(float4 rgba, int pf)
{
    int arr = pf & PIXEL_ARR_MASK;

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
        return (float4)(
            0.5f * r - 0.418688f * g - 0.081312f * b,
            -0.168736f * r - 0.331264f * g + 0.5f * b,
            0.299f * r + 0.587f * g + 0.114f * b,
            a
        );
    }
}

// Write float4 (storage-native) to byte buffer
static inline void WriteRawPixelOCL(__global uchar* base, uint byteOff, float4 pixel, int pf)
{
    int type = pf & PIXEL_TYPE_MASK;

    if (type == PIXEL_TYPE_F32)
    {
        pixel.w = Clamp01(pixel.w);
        vstore4(pixel, 0, (__global float*)(base + byteOff));
    }
    else if (type == PIXEL_TYPE_F16)
    {
        // float4 -> half4
        __global half* p = (__global half*)(base + byteOff);
        p[0] = (half)pixel.x;
        p[1] = (half)pixel.y;
        p[2] = (half)pixel.z;
        p[3] = (half)pixel.w;
    }
    else if (type == PIXEL_TYPE_U16)
    {
        ushort4 q = convert_ushort4_sat(pixel * 32767.5f);
        __global ushort* p = (__global ushort*)(base + byteOff);
        p[0] = q.x; p[1] = q.y; p[2] = q.z; p[3] = q.w;
    }
    else // PIXEL_TYPE_U8
    {
        uchar4 q = convert_uchar4_sat(pixel * 255.0f);
        vstore4(q, 0, base + byteOff);
    }
}

// Convenience: read + convert to RGBA
static inline float4 LoadPixelRGBAOCL(__global const uchar* base, uint byteOff, int pf)
{
    return StorageToRGBAOCL(ReadRawPixelOCL(base, byteOff, pf), pf);
}

// Convenience: convert RGBA + write
static inline void StorePixelRGBAOCL(__global uchar* base, uint byteOff, float4 rgba, int pf)
{
    WriteRawPixelOCL(base, byteOff, RGBAToStorageOCL(rgba, pf), pf);
}

// ============================================================================
// Blend all samples for one pixel
// ============================================================================
static inline float4 BlendSamplesOCL(
    __global const uchar* samples, uint rowByteOff,
    __global const float* opacities,
    int pf, int numSamples, int blendMode,
    int sampleImageSize)
{
    int ps = PixelSizeOCL(pf);
    float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    for (int i = 0; i < numSamples; ++i)
    {
        uint srcOff = i * sampleImageSize + rowByteOff;
        float4 sample = LoadPixelRGBAOCL(samples, srcOff, pf);
        float4 premul = PremultiplyKernel(sample, opacities[i]);

        if (blendMode == BLEND_NEW_ON_TOP)
            acc = BlendNewOnTopKernel(acc, premul);
        else if (blendMode == BLEND_NEW_ON_BOTTOM)
            acc = BlendNewOnBottomKernel(acc, premul);
        else if (blendMode == BLEND_ADD)
            acc = BlendAddKernel(acc, premul);
        else if (blendMode == BLEND_MAXIMUM)
            acc = BlendMaximumKernel(acc, premul);
        else
            acc = BlendMinimumKernel(acc, premul);
    }

    return UnpremultiplyKernel(acc);
}

// ============================================================================
// Path 1: Premiere Pro GPU Filter Framework (GF_KERNEL_FUNCTION macro)
//
// NOTE: The Pr framework kernel support does not define uchar/half types.
// The host converts all sample data to float4 RGBA before uploading, and
// the output is also float4.  Format conversion happens on the CPU host side.
// ============================================================================

// -----------------------------------------------------------------------
// Float32/16 hybrid kernel (via ReadFloat4/WriteFloat4 from KernelMemory.h)
// When inIs16Bit=0:  all data is float4 (32f path) - direct access
// When inIs16Bit=1:  buffers contain half4 (16f path) - ReadFloat4/WriteFloat4
//                     convert transparently; compute stays in float32.
// -----------------------------------------------------------------------
GF_KERNEL_FUNCTION(FrameEchoBlendKernel,
    ((GF_PTR(float4))(ioImage))
    ((GF_PTR(const float4))(inSamples))
    ((GF_PTR(const float))(inOpacities)),
    ((int)(inBoxLeft))
    ((int)(inBoxTop))
    ((int)(inBoxWidth))
    ((int)(inBoxHeight))
    ((int)(inBytePitch))   // row stride in float4 units (= bytes / 16)
    ((int)(inHeight))
    ((int)(inNumSamples))
    ((int)(inBlendMode))
    ((int)(inIs16Bit)),    // 0=float4 buffers, 1=half4 buffers with auto-convert
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inBoxWidth && inXY.y < inBoxHeight)
    {
        uint2 px = (uint2)(inBoxLeft + inXY.x, inBoxTop + inXY.y);
        uint sampleImagePitch = inBytePitch * inHeight; // elements per sample image
        uint rowOff = px.y * inBytePitch + px.x;        // element index

        float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

        for (int i = 0; i < inNumSamples; ++i)
        {
            float4 sample = ReadFloat4(inSamples, i * sampleImagePitch + rowOff, (bool)inIs16Bit);
            float4 premul = PremultiplyKernel(sample, inOpacities[i]);

            if (inBlendMode == BLEND_NEW_ON_TOP)
                acc = BlendNewOnTopKernel(acc, premul);
            else if (inBlendMode == BLEND_NEW_ON_BOTTOM)
                acc = BlendNewOnBottomKernel(acc, premul);
            else if (inBlendMode == BLEND_ADD)
                acc = BlendAddKernel(acc, premul);
            else if (inBlendMode == BLEND_MAXIMUM)
                acc = BlendMaximumKernel(acc, premul);
            else
                acc = BlendMinimumKernel(acc, premul);
        }

        WriteFloat4(UnpremultiplyKernel(acc), ioImage, rowOff, (bool)inIs16Bit);
    }
}

// ============================================================================
// Path 2: Standalone OpenCL entry point (always compiled)
// Used by Ae GPU path. Supports all pixel formats via inPixelFormat.
// ============================================================================

__kernel void FrameEchoBlend_OCL(
    __global uchar* ioImage,
    __global const uchar* inSamples,
    __global const float* inOpacities,
    int inBoxLeft,
    int inBoxTop,
    int inBoxWidth,
    int inBoxHeight,
    int inBytePitch,
    int inHeight,
    int inPixelFormat,
    int inNumSamples,
    int inBlendMode)
{
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    if (gx >= inBoxWidth || gy >= inBoxHeight) return;

    uint2 px = (uint2)(inBoxLeft + gx, inBoxTop + gy);
    uint sampleImageSize = inBytePitch * inHeight;
    uint rowByteOff = px.y * inBytePitch + px.x * PixelSizeOCL(inPixelFormat);

    float4 result = BlendSamplesOCL(
        inSamples, rowByteOff, inOpacities,
        inPixelFormat, inNumSamples, inBlendMode,
        sampleImageSize);

    StorePixelRGBAOCL(ioImage, rowByteOff, result, inPixelFormat);
}

#endif // FRAME_ECHO_BLEND_CL
