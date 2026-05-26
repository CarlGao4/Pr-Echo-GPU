//=============================================================================
// FrameEchoBlend.cu - CUDA Kernel
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
// Compiled by CMake's native CUDA support (enable_language(CUDA)).
// The __global__ kernel is statically linked into the final .aex binary.
// CMAKE_CUDA_ARCHITECTURES controls which GPU architectures are targeted.
//=============================================================================

#include <cuda_runtime.h>
#include <cuda.h>          // CUDA Driver API types for host wrapper
#include <cuda_fp16.h>     // __half, __half2float, __float2half
#include <device_launch_parameters.h>
#include <cstdint>

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
// Device helpers
// ============================================================================

__device__ __forceinline__ float Clamp01(float v)
{
    return fmaxf(0.0f, fminf(1.0f, v));
}

__device__ __forceinline__ float4 PremultiplyKernel(float4 pixel, float opacity)
{
    float alpha = Clamp01(pixel.w * opacity);
    return make_float4(pixel.x * alpha, pixel.y * alpha, pixel.z * alpha, alpha);
}

__device__ __forceinline__ float4 UnpremultiplyKernel(float4 premul)
{
    if (premul.w <= 0.0f) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float clamped = Clamp01(premul.w);
    float inv = 1.0f / clamped;
    return make_float4(premul.x * inv, premul.y * inv, premul.z * inv, clamped);
}

// ============================================================================
// Five blend functions - operate on float4 RGBA only
// ============================================================================

__device__ __forceinline__ float4 BlendNewOnTopKernel(float4 acc, float4 premul)
{
    float omsa = 1.0f - premul.w;
    return make_float4(
        premul.x + acc.x * omsa,
        premul.y + acc.y * omsa,
        premul.z + acc.z * omsa,
        premul.w + acc.w * omsa
    );
}

__device__ __forceinline__ float4 BlendNewOnBottomKernel(float4 acc, float4 premul)
{
    float omsa = 1.0f - acc.w;
    return make_float4(
        acc.x + premul.x * omsa,
        acc.y + premul.y * omsa,
        acc.z + premul.z * omsa,
        acc.w + premul.w * omsa
    );
}

__device__ __forceinline__ float4 BlendAddKernel(float4 acc, float4 premul)
{
    return make_float4(
        acc.x + premul.x, acc.y + premul.y,
        acc.z + premul.z, acc.w + premul.w
    );
}

__device__ __forceinline__ float4 BlendMaximumKernel(float4 acc, float4 premul)
{
    return make_float4(
        fmaxf(acc.x, premul.x), fmaxf(acc.y, premul.y),
        fmaxf(acc.z, premul.z), fmaxf(acc.w, premul.w)
    );
}

__device__ __forceinline__ float4 BlendMinimumKernel(float4 acc, float4 premul)
{
    return make_float4(
        fminf(acc.x, premul.x), fminf(acc.y, premul.y),
        fminf(acc.z, premul.z), fminf(acc.w, premul.w)
    );
}

// ============================================================================
// Pixel format conversion helpers
// ============================================================================

__device__ __forceinline__ int PixelSizeCUDA(int pf)
{
    int type = pf & PIXEL_TYPE_MASK;
    if (type == PIXEL_TYPE_U8)  return 4;
    if (type == PIXEL_TYPE_U16) return 8;
    if (type == PIXEL_TYPE_F16) return 8;
    return 16;
}

__device__ __forceinline__ float4 ReadRawPixelCUDA(const uint8_t* base, uint32_t byteOff, int pf)
{
    int type = pf & PIXEL_TYPE_MASK;

    if (type == PIXEL_TYPE_F32)
    {
        return *(const float4*)(base + byteOff);
    }
    else if (type == PIXEL_TYPE_F16)
    {
        // half4 -> float4 using __half2float
        const __half* p = (const __half*)(base + byteOff);
        return make_float4(
            __half2float(p[0]), __half2float(p[1]),
            __half2float(p[2]), __half2float(p[3])
        );
    }
    else if (type == PIXEL_TYPE_U16)
    {
        const uint16_t* p = (const uint16_t*)(base + byteOff);
        return make_float4(
            p[0] * (1.0f / 32768.0f), p[1] * (1.0f / 32768.0f),
            p[2] * (1.0f / 32768.0f), p[3] * (1.0f / 32768.0f)
        );
    }
    else // PIXEL_TYPE_U8
    {
        uint32_t raw = *(const uint32_t*)(base + byteOff);
        const float k = 1.0f / 255.0f;
        return make_float4(
            (raw & 0xFF) * k, ((raw >> 8) & 0xFF) * k,
            ((raw >> 16) & 0xFF) * k, ((raw >> 24) & 0xFF) * k
        );
    }
}

// Storage-native float4 -> standard float4 RGBA
__device__ __forceinline__ float4 StorageToRGBA(float4 pixel, int pf)
{
    int arr = pf & PIXEL_ARR_MASK;

    if (arr == PIXEL_ARR_BGRA) return make_float4(pixel.z, pixel.y, pixel.x, pixel.w);
    if (arr == PIXEL_ARR_ARGB) return make_float4(pixel.y, pixel.z, pixel.w, pixel.x);

    // VUYA
    float v = pixel.x, u = pixel.y, y = pixel.z, a = pixel.w;
    return make_float4(
        y + 1.402f * v,
        y - 0.344136f * u - 0.714136f * v,
        y + 1.772f * u,
        a
    );
}

// float4 RGBA -> storage-native float4
__device__ __forceinline__ float4 RGBAToStorage(float4 rgba, int pf)
{
    int arr = pf & PIXEL_ARR_MASK;

    if (arr == PIXEL_ARR_BGRA) return make_float4(rgba.z, rgba.y, rgba.x, rgba.w);
    if (arr == PIXEL_ARR_ARGB) return make_float4(rgba.w, rgba.x, rgba.y, rgba.z);

    // VUYA
    float r = rgba.x, g = rgba.y, b = rgba.z, a = rgba.w;
    return make_float4(
        0.5f * r - 0.418688f * g - 0.081312f * b,
        -0.168736f * r - 0.331264f * g + 0.5f * b,
        0.299f * r + 0.587f * g + 0.114f * b,
        a
    );
}

// Write float4 (storage-native) to byte buffer, quantizing as needed
__device__ __forceinline__ void WriteRawPixelCUDA(uint8_t* base, uint32_t byteOff, float4 pixel, int pf)
{
    int type = pf & PIXEL_TYPE_MASK;

    if (type == PIXEL_TYPE_F32)
    {
        pixel.w = Clamp01(pixel.w);
        *(float4*)(base + byteOff) = pixel;
    }
    else if (type == PIXEL_TYPE_F16)
    {
        // float4 -> half4 using __float2half
        __half* p = (__half*)(base + byteOff);
        p[0] = __float2half(pixel.x);
        p[1] = __float2half(pixel.y);
        p[2] = __float2half(pixel.z);
        p[3] = __float2half(pixel.w);
    }
    else if (type == PIXEL_TYPE_U16)
    {
        uint16_t* p = (uint16_t*)(base + byteOff);
        p[0] = (uint16_t)__float2uint_rn(fminf(fmaxf(pixel.x, 0.0f), 1.0f) * 32767.5f);
        p[1] = (uint16_t)__float2uint_rn(fminf(fmaxf(pixel.y, 0.0f), 1.0f) * 32767.5f);
        p[2] = (uint16_t)__float2uint_rn(fminf(fmaxf(pixel.z, 0.0f), 1.0f) * 32767.5f);
        p[3] = (uint16_t)__float2uint_rn(fminf(fmaxf(pixel.w, 0.0f), 1.0f) * 32767.5f);
    }
    else // PIXEL_TYPE_U8
    {
        uint32_t raw =
            ((uint32_t)__float2uint_rn(fminf(fmaxf(pixel.x, 0.0f), 1.0f) * 255.0f) & 0xFF) |
            (((uint32_t)__float2uint_rn(fminf(fmaxf(pixel.y, 0.0f), 1.0f) * 255.0f) & 0xFF) << 8) |
            (((uint32_t)__float2uint_rn(fminf(fmaxf(pixel.z, 0.0f), 1.0f) * 255.0f) & 0xFF) << 16) |
            (((uint32_t)__float2uint_rn(fminf(fmaxf(pixel.w, 0.0f), 1.0f) * 255.0f) & 0xFF) << 24);
        *(uint32_t*)(base + byteOff) = raw;
    }
}

// Convenience: read + convert to RGBA
__device__ __forceinline__ float4 LoadPixelRGBA(const uint8_t* base, uint32_t byteOff, int pf)
{
    return StorageToRGBA(ReadRawPixelCUDA(base, byteOff, pf), pf);
}

// Convenience: convert RGBA + write
__device__ __forceinline__ void StorePixelRGBA(uint8_t* base, uint32_t byteOff, float4 rgba, int pf)
{
    WriteRawPixelCUDA(base, byteOff, RGBAToStorage(rgba, pf), pf);
}

// ============================================================================
// Blend all samples for one pixel
// ============================================================================
__device__ __forceinline__ float4 BlendSamplesCUDA(
    const uint8_t* samples, uint32_t rowByteOff,
    const float*   opacities,
    int pf, int numSamples, int blendMode,
    uint32_t sampleImageSize)
{
    uint32_t ps = PixelSizeCUDA(pf);
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int i = 0; i < numSamples; ++i)
    {
        uint32_t srcOff = (uint32_t)i * sampleImageSize + rowByteOff;
        float4 sample = LoadPixelRGBA(samples, srcOff, pf);
        float4 premul = PremultiplyKernel(sample, opacities[i]);

        switch (blendMode)
        {
        case BLEND_NEW_ON_TOP:    acc = BlendNewOnTopKernel(acc, premul);    break;
        case BLEND_NEW_ON_BOTTOM: acc = BlendNewOnBottomKernel(acc, premul); break;
        case BLEND_ADD:           acc = BlendAddKernel(acc, premul);         break;
        case BLEND_MAXIMUM:       acc = BlendMaximumKernel(acc, premul);     break;
        default:                  acc = BlendMinimumKernel(acc, premul);     break;
        }
    }

    return UnpremultiplyKernel(acc);
}

// ============================================================================
// Main composition kernel
//
// Dispatch grid = ceil(inBoxWidth/16) * ceil(inBoxHeight/16)
// Each thread maps directly into the box region.
// ============================================================================

__global__ void FrameEchoBlend_CUDA(
    uint8_t*       ioImage,
    const uint8_t* inSamples,
    const float*   inOpacities,
    int            inBoxLeft,
    int            inBoxTop,
    int            inBoxWidth,
    int            inBoxHeight,
    int            inBytePitch,
    int            inHeight,
    int            inPixelFormat,
    int            inNumSamples,
    int            inBlendMode)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= inBoxWidth || gy >= inBoxHeight) return;

    uint32_t px_x = inBoxLeft + gx;
    uint32_t px_y = inBoxTop  + gy;
    uint32_t sampleImageSize = inBytePitch * inHeight;
    uint32_t rowByteOff = px_y * inBytePitch + px_x * PixelSizeCUDA(inPixelFormat);

    float4 result = BlendSamplesCUDA(
        inSamples, rowByteOff, inOpacities,
        inPixelFormat, inNumSamples, inBlendMode,
        sampleImageSize);

    StorePixelRGBA(ioImage, rowByteOff, result, inPixelFormat);
}

// ============================================================================
// Native half4 composition kernel
//
// Processes all blend operations entirely in half precision for GPUs with
// native float16 support (Compute Capability >= 6.0).
// Dispatch grid = ceil(inBoxWidth/16) * ceil(inBoxHeight/16)
// ============================================================================

// Custom half4 struct matching buffer layout (4 contiguous __half values)
struct alignas(8) Half4
{
    __half x, y, z, w;
};

__device__ __forceinline__ Half4 PremultiplyHalf(Half4 pixel, float opacity)
{
    float a_f = __half2float(pixel.w) * opacity;
    a_f = fmaxf(0.0f, fminf(1.0f, a_f));
    __half a = __float2half(a_f);
    Half4 result;
    result.x = __hmul(pixel.x, a);
    result.y = __hmul(pixel.y, a);
    result.z = __hmul(pixel.z, a);
    result.w = a;
    return result;
}

__device__ __forceinline__ Half4 UnpremultiplyHalf(Half4 premul)
{
    float alpha_f = __half2float(premul.w);
    if (alpha_f <= 0.0f)
    {
        Half4 zero; zero.x = zero.y = zero.z = zero.w = __float2half(0.0f);
        return zero;
    }
    float clamped = fmaxf(0.0f, fminf(1.0f, alpha_f));
    float invF = 1.0f / clamped;
    __half inv = __float2half(invF);
    __half clampedH = __float2half(clamped);
    Half4 result;
    result.x = __hmul(premul.x, inv);
    result.y = __hmul(premul.y, inv);
    result.z = __hmul(premul.z, inv);
    result.w = clampedH;
    return result;
}

// Blend operations in half precision
__device__ __forceinline__ Half4 BlendNewOnTopHalf(Half4 acc, Half4 premul)
{
    Half4 one; one.x = one.y = one.z = one.w = __float2half(1.0f);
    // omsa = 1.0 - premul.w
    __half omsa = __hsub(one.w, premul.w);
    Half4 r;
    r.x = __hadd(premul.x, __hmul(acc.x, omsa));
    r.y = __hadd(premul.y, __hmul(acc.y, omsa));
    r.z = __hadd(premul.z, __hmul(acc.z, omsa));
    r.w = __hadd(premul.w, __hmul(acc.w, omsa));
    return r;
}

__device__ __forceinline__ Half4 BlendNewOnBottomHalf(Half4 acc, Half4 premul)
{
    Half4 one; one.x = one.y = one.z = one.w = __float2half(1.0f);
    // omsa = 1.0 - acc.w
    __half omsa = __hsub(one.w, acc.w);
    Half4 r;
    r.x = __hadd(acc.x, __hmul(premul.x, omsa));
    r.y = __hadd(acc.y, __hmul(premul.y, omsa));
    r.z = __hadd(acc.z, __hmul(premul.z, omsa));
    r.w = __hadd(acc.w, __hmul(premul.w, omsa));
    return r;
}

__device__ __forceinline__ Half4 BlendAddHalf(Half4 acc, Half4 premul)
{
    Half4 r;
    r.x = __hadd(acc.x, premul.x);
    r.y = __hadd(acc.y, premul.y);
    r.z = __hadd(acc.z, premul.z);
    r.w = __hadd(acc.w, premul.w);
    return r;
}

__device__ __forceinline__ Half4 BlendMaximumHalf(Half4 acc, Half4 premul)
{
    Half4 r;
    r.x = __hmax(acc.x, premul.x);
    r.y = __hmax(acc.y, premul.y);
    r.z = __hmax(acc.z, premul.z);
    r.w = __hmax(acc.w, premul.w);
    return r;
}

__device__ __forceinline__ Half4 BlendMinimumHalf(Half4 acc, Half4 premul)
{
    Half4 r;
    r.x = __hmin(acc.x, premul.x);
    r.y = __hmin(acc.y, premul.y);
    r.z = __hmin(acc.z, premul.z);
    r.w = __hmin(acc.w, premul.w);
    return r;
}

__global__ void FrameEchoBlend_CUDA_Half(
    Half4*         ioImage,
    const Half4*   inSamples,
    const float*   inOpacities,
    int            inBoxLeft,
    int            inBoxTop,
    int            inBoxWidth,
    int            inBoxHeight,
    int            inHalf4Pitch,   // row stride in half4 elements (= rowBytes / 8)
    int            inHeight,
    int            inNumSamples,
    int            inBlendMode)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= inBoxWidth || gy >= inBoxHeight) return;

    uint32_t px_x = inBoxLeft + gx;
    uint32_t px_y = inBoxTop  + gy;
    uint32_t sampleImagePitch = inHalf4Pitch * inHeight;
    uint32_t rowOff = px_y * inHalf4Pitch + px_x;

    Half4 acc;
    acc.x = acc.y = acc.z = acc.w = __float2half(0.0f);

    for (int i = 0; i < inNumSamples; ++i)
    {
        Half4 sample = inSamples[i * sampleImagePitch + rowOff];
        Half4 premul = PremultiplyHalf(sample, inOpacities[i]);

        switch (inBlendMode)
        {
        case BLEND_NEW_ON_TOP:    acc = BlendNewOnTopHalf(acc, premul);    break;
        case BLEND_NEW_ON_BOTTOM: acc = BlendNewOnBottomHalf(acc, premul); break;
        case BLEND_ADD:           acc = BlendAddHalf(acc, premul);         break;
        case BLEND_MAXIMUM:       acc = BlendMaximumHalf(acc, premul);     break;
        default:                  acc = BlendMinimumHalf(acc, premul);     break;
        }
    }

    ioImage[rowOff] = UnpremultiplyHalf(acc);
}

// ============================================================================
// Host-side launch wrapper (extern "C" for MSVC callers)
//
// Static linking: the __global__ kernel is compiled and linked directly into
// the .aex via CMake's enable_language(CUDA). No cuModuleLoad needed.
// Accepts CUDA Driver API handles (CUdeviceptr / CUstream) which come from
// Pr's PrGPUDeviceInfo or AE's GPU Device Suite.
// ============================================================================

extern "C" {

void FrameEchoBlend_CUDA_Launch(
    CUdeviceptr    ioImage,
    CUdeviceptr    inSamples,
    CUdeviceptr    inOpacities,
    int            inBoxLeft,
    int            inBoxTop,
    int            inBoxWidth,
    int            inBoxHeight,
    int            inBytePitch,
    int            inHeight,
    int            inPixelFormat,
    int            inNumSamples,
    int            inBlendMode,
    CUstream       stream)
{
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (inBoxWidth  + blockDim.x - 1) / blockDim.x,
        (inBoxHeight + blockDim.y - 1) / blockDim.y,
        1);

    FrameEchoBlend_CUDA<<<gridDim, blockDim, 0, stream>>>(
        reinterpret_cast<uint8_t*>(ioImage),
        reinterpret_cast<const uint8_t*>(inSamples),
        reinterpret_cast<const float*>(inOpacities),
        inBoxLeft, inBoxTop,
        inBoxWidth, inBoxHeight,
        inBytePitch, inHeight,
        inPixelFormat,
        inNumSamples, inBlendMode);
}

void FrameEchoBlend_CUDA_Half_Launch(
    CUdeviceptr    ioImage,
    CUdeviceptr    inSamples,
    CUdeviceptr    inOpacities,
    int            inBoxLeft,
    int            inBoxTop,
    int            inBoxWidth,
    int            inBoxHeight,
    int            inHalf4Pitch,   // row stride in half4 elements (= rowBytes / 8)
    int            inHeight,
    int            inNumSamples,
    int            inBlendMode,
    CUstream       stream)
{
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (inBoxWidth  + blockDim.x - 1) / blockDim.x,
        (inBoxHeight + blockDim.y - 1) / blockDim.y,
        1);

    FrameEchoBlend_CUDA_Half<<<gridDim, blockDim, 0, stream>>>(
        reinterpret_cast<Half4*>(ioImage),
        reinterpret_cast<const Half4*>(inSamples),
        reinterpret_cast<const float*>(inOpacities),
        inBoxLeft, inBoxTop,
        inBoxWidth, inBoxHeight,
        inHalf4Pitch, inHeight,
        inNumSamples, inBlendMode);
}

} // extern "C"
