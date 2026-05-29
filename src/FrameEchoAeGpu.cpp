//=============================================================================
// FrameEchoAeGpu.cpp - After Effects Smart Render GPU implementation
//
// Handles:
//   AeGPUDeviceSetup()    - compile OpenCL kernel / prepare CUDA
//   AeGPUDeviceSetdown()  - release per-device resources
//   AeSmartRenderGPU()    - assemble frames, upload to GPU, launch kernel
//
// The GPU kernel (FrameEchoBlend_OCL / FrameEchoBlend_CUDA) handles all pixel
// format conversions internally. The same preprocessed OpenCL string
// (kFrameEchoBlend_OpenCLString) is used for both hosts.
//=============================================================================

#include "FrameEchoAeSmartRender.h"
#include "FrameEchoCore.h"
#include "FrameEchoParams.h"

#include "AEConfig.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_EffectGPUSuites.h"
#include "AE_Macros.h"
#include "AEFX_SuiteHandlerTemplate.h"

#include "PrSDKAESupport.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

// ============================================================================
// GPU backend support
// ============================================================================

#ifdef GPU_ENABLE_OPENCL
#  include "FrameEchoBlend.cl.h"
#  include <CL/cl.h>

// Convert OpenCL error to PF_Err (from SDK sample)
inline PF_Err CL2Err(cl_int cl_result) {
    if (cl_result == CL_SUCCESS) return PF_Err_NONE;
    return PF_Err_INTERNAL_STRUCT_DAMAGED;
}
#  define CL_ERR(FUNC) ERR(CL2Err(FUNC))
#endif

#ifdef GPU_ENABLE_CUDA
#  include <cuda.h>

   // Launch wrappers from FrameEchoBlend.cu (statically linked by nvcc)
   extern "C" void FrameEchoBlend_CUDA_Launch(
       CUdeviceptr ioImage, CUdeviceptr inSamples, CUdeviceptr inOpacities,
       int inBoxLeft, int inBoxTop, int inBoxWidth, int inBoxHeight,
       int inBytePitch, int inHeight, int inPixelFormat,
       int inNumSamples, int inBlendMode, CUstream stream);

   extern "C" void FrameEchoBlend_CUDA_Half_Launch(
       CUdeviceptr ioImage, CUdeviceptr inSamples, CUdeviceptr inOpacities,
       int inBoxLeft, int inBoxTop, int inBoxWidth, int inBoxHeight,
       int inHalf4Pitch, int inHeight,
       int inNumSamples, int inBlendMode, CUstream stream);
#endif

// ============================================================================
// Per-device cached resources (indexed by device_index)
// ============================================================================
enum { kMaxAeDevices = 12 };

#ifdef GPU_ENABLE_OPENCL
static cl_kernel sAeOpenCLKernel[kMaxAeDevices] = {};
#endif

// ============================================================================
// Ae GPU pixel format -> kernel pixelFormatBits mapping
// ============================================================================
#define PIXEL_ARR_BGRA  1
#define PIXEL_ARR_ARGB  2
#define PIXEL_TYPE_U8    (0 << 5)
#define PIXEL_TYPE_U16   (1 << 5)
#define PIXEL_TYPE_F32   (2 << 5)
#define PIXEL_TYPE_F16   (3 << 5)

struct AePixelFormatInfo
{
    int pixelFormatBits;
    int elementSize;  // bytes per pixel
    int pixelType;    // raw type bits (0xE0 mask)
};

static AePixelFormatInfo MapAePixelFormat(PF_PixelFormat pf)
{
    switch (pf)
    {
    case PF_PixelFormat_GPU_BGRA128:
        return { PIXEL_ARR_BGRA | PIXEL_TYPE_F32, 16, PIXEL_TYPE_F32 };
    case PF_PixelFormat_ARGB128:
        return { PIXEL_ARR_ARGB | PIXEL_TYPE_F32, 16, PIXEL_TYPE_F32 };
    case PF_PixelFormat_ARGB64:
        return { PIXEL_ARR_ARGB | PIXEL_TYPE_U16, 8, PIXEL_TYPE_U16 };
    case PF_PixelFormat_ARGB32:
        return { PIXEL_ARR_ARGB | PIXEL_TYPE_U8, 4, PIXEL_TYPE_U8 };
    default:
        return { PIXEL_ARR_BGRA | PIXEL_TYPE_F32, 16, PIXEL_TYPE_F32 };
    }
}

// Host-side float32 -> float16 (IEEE 754) conversion for SolidColor fills
static inline uint16_t Float32ToFloat16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x007FFFFF;

    if (exp <= 0)
    {
        return (uint16_t)sign;
    }
    if (exp >= 31)
    {
        return (uint16_t)(sign | 0x7C00 | (mant ? 0x0200 : 0));
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

// ============================================================================
// Blend mode -> kernel enum (same values as Pr GPU path)
// ============================================================================
static int BlendModeToKernelEnum(frame_echo::BlendMode mode)
{
    using namespace frame_echo;
    switch (mode)
    {
    case BlendMode::BlendNewOnTop:    return 0;
    case BlendMode::BlendNewOnBottom: return 1;
    case BlendMode::Add:              return 2;
    case BlendMode::Maximum:          return 3;
    case BlendMode::Minimum:          return 4;
    default:                          return 0;
    }
}

// ============================================================================
// AeGPUDeviceSetup
//
// Compiles FrameEchoBlend_OCL on OpenCL devices.
// For CUDA the kernel is statically linked - nothing to do.
// ============================================================================
PF_Err AeGPUDeviceSetup(
    PF_InData*      in_data,
    PF_OutData*     out_data,
    PF_GPUDeviceSetupExtra* extra)
{
    PF_Err err = PF_Err_NONE;

    PF_GPUDeviceInfo deviceInfo;
    AEFX_CLR_STRUCT(deviceInfo);

    AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpuDeviceSuite(
        in_data, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_data);

    ERR(gpuDeviceSuite->GetDeviceInfo(
        in_data->effect_ref, extra->input->device_index, &deviceInfo));

#ifdef GPU_ENABLE_OPENCL
    if (extra->input->what_gpu == PF_GPU_Framework_OPENCL)
    {
        // Return cached kernel if already compiled for this device
        if (sAeOpenCLKernel[extra->input->device_index])
        {
            extra->output->gpu_data = nullptr; // no per-instance data needed
            out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
            return PF_Err_NONE;
        }

        cl_int result = CL_SUCCESS;
        cl_context context   = (cl_context)deviceInfo.contextPV;
        cl_device_id device = (cl_device_id)deviceInfo.devicePV;

        // Prepend the 16f support define (0 = Ae uses 32f only)
        const char* k16fString = "#define GF_OPENCL_SUPPORTS_16F 0\n";
        size_t sizes[]  = { strlen(k16fString), kFrameEchoBlend_OpenCLString_Size };
        const char* strings[] = { k16fString, reinterpret_cast<const char*>(kFrameEchoBlend_OpenCLString) };

        cl_program program = clCreateProgramWithSource(
            context, 2, &strings[0], &sizes[0], &result);
        CL_ERR(result);

        if (!err)
        {
            result = clBuildProgram(program, 1, &device,
                "-cl-single-precision-constant -cl-fast-relaxed-math",
                nullptr, nullptr);
            if (result != CL_SUCCESS)
            {
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }
        }

        if (!err)
        {
            sAeOpenCLKernel[extra->input->device_index] =
                clCreateKernel(program, "FrameEchoBlend_OCL", &result);
            CL_ERR(result);
        }

        // Release program (kernel keeps a reference internally)
        clReleaseProgram(program);

        // No per-instance data - we use the global cache
        extra->output->gpu_data = nullptr;

        if (!err)
        {
            out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
        }
    }
#endif

#ifdef GPU_ENABLE_CUDA
    if (extra->input->what_gpu == PF_GPU_Framework_CUDA)
    {
        // CUDA kernel is statically linked - no runtime setup needed.
        extra->output->gpu_data = nullptr;
        out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
#endif

    return err;
}

// ============================================================================
// AeGPUDeviceSetdown
// ============================================================================
PF_Err AeGPUDeviceSetdown(
    PF_InData*      in_data,
    PF_OutData*     out_data,
    PF_GPUDeviceSetdownExtra* extra)
{
#ifdef GPU_ENABLE_OPENCL
    if (extra->input->what_gpu == PF_GPU_Framework_OPENCL)
    {
        csSDK_int32 idx = extra->input->device_index;
        if (idx >= 0 && idx < kMaxAeDevices && sAeOpenCLKernel[idx])
        {
            clReleaseKernel(sAeOpenCLKernel[idx]);
            sAeOpenCLKernel[idx] = nullptr;
        }
    }
#endif
    return PF_Err_NONE;
}

// ============================================================================
// AeSmartRenderGPU
//
// Uses AeSmartRenderAssemble (from FrameEchoSmartRender.cpp) for shared
// checkout, pixel format detection, dimension validation, and repeat.
// Uploads frames to GPU and launches the blend kernel.
// The kernel (FrameEchoBlend_OCL / FrameEchoBlend_CUDA) handles all pixel
// format conversions internally via inPixelFormat.
// ============================================================================
PF_Err AeSmartRenderGPU(
    PF_InData*           in_data,
    PF_OutData*          out_data,
    PF_SmartRenderExtra* extra,
    AePreRenderData*     aeData)
{
    PF_Err err = PF_Err_NONE;

    if (!aeData) return PF_Err_INTERNAL_STRUCT_DAMAGED;

    // ---- Shared assembly: checkout, pixel format, validation, repeat ----
    AeSmartRenderPrep prep;
    ERR(AeSmartRenderAssemble(in_data, out_data, extra, aeData, prep));
    if (err)
    {
        AeSmartRenderCheckin(in_data, extra, prep);
        return err;
    }

    const int nb = aeData->backwardCount;
    const int nf = aeData->forwardCount;
    const int numSamples = prep.numSamples;
    const int width = prep.width;
    const int height = prep.height;
    const frame_echo::TemporalSettings& settings = aeData->settings;

    // ---- Map pixel format for kernel ----
    auto pfInfo = MapAePixelFormat(prep.pixelFormat);
    const int elementPitch = prep.outputWorld->rowbytes / pfInfo.elementSize;
    const size_t sampleImageSize = (size_t)elementPitch * (size_t)height;
    const size_t sampleByteSize  = sampleImageSize * pfInfo.elementSize;
    const size_t totalBytes = (size_t)numSamples * sampleByteSize;

    // ---- Get GPU device info and suites ----
    AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpuSuite(
        in_data, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_data);

    PF_GPUDeviceInfo deviceInfo;
    AEFX_CLR_STRUCT(deviceInfo);
    ERR(gpuSuite->GetDeviceInfo(in_data->effect_ref, extra->input->device_index, &deviceInfo));

    // ---- Retrieve output GPU pointer ----
    void* outDevPtr = nullptr;
    ERR(gpuSuite->GetGPUWorldData(in_data->effect_ref, prep.outputWorld, &outDevPtr));
    if (!outDevPtr)
    {
        AeSmartRenderCheckin(in_data, extra, prep);
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    // ---- Build devPtrs from resolvedSlot ----
    std::vector<void*> devPtrs(numSamples, nullptr);
    for (const auto& cf : prep.checkedFrames)
    {
        if (cf.resolvedSlot >= 0 && cf.resolvedSlot < numSamples && cf.world)
        {
            void* dp = nullptr;
            ERR(gpuSuite->GetGPUWorldData(in_data->effect_ref, cf.world, &dp));
            devPtrs[cf.resolvedSlot] = dp;
        }
    }

    // ========================================================================
    // Upload to GPU buffer + handle shortage + launch kernel
    // ========================================================================

    // ------------------------------------------------------------------
    // OpenCL path
    // ------------------------------------------------------------------
    if (!err && extra->input->what_gpu == PF_GPU_Framework_OPENCL)
    {
#ifdef GPU_ENABLE_OPENCL
        cl_command_queue cq = (cl_command_queue)deviceInfo.command_queuePV;
        cl_context ctx      = (cl_context)deviceInfo.contextPV;
        cl_int res;

        cl_mem samplesBuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                           totalBytes, nullptr, &res);
        CL_ERR(res);

        // Pre-fill samples buffer with shortage pattern (BlankTransparent / SolidColor)
        if (!err)
        {
            const auto& shortage = settings.frameShortageBehavior;
            if (shortage == frame_echo::FrameShortageBehavior::BlankTransparent)
            {
                cl_int zero = 0;
                for (int s = 0; s < numSamples; ++s)
                {
                    if (s == nb) continue;
                    if (devPtrs[s]) continue;
                    res = clEnqueueFillBuffer(cq, samplesBuf, &zero, sizeof(cl_int),
                        (size_t)s * sampleByteSize, sampleByteSize, 0, nullptr, nullptr);
                    CL_ERR(res);
                }
            }
            else if (shortage == frame_echo::FrameShortageBehavior::SolidColor)
            {
                // Build fill buffer in the native pixel layout
                const int arr = pfInfo.pixelFormatBits & 0x1F;
                std::vector<uint8_t> fillBuf(sampleByteSize);

                if (pfInfo.elementSize == 16)
                {
                    // float4 layout
                    float* fp = reinterpret_cast<float*>(fillBuf.data());
                    const float b = settings.shortageColor.b;
                    const float g = settings.shortageColor.g;
                    const float r = settings.shortageColor.r;
                    const float a = settings.shortageColor.a;
                    for (size_t px = 0; px < sampleImageSize; ++px)
                    {
                        if (arr == 2 /*PIXEL_ARR_ARGB*/)
                        {
                            fp[px * 4 + 0] = a; fp[px * 4 + 1] = r;
                            fp[px * 4 + 2] = g; fp[px * 4 + 3] = b;
                        }
                        else // BGRA
                        {
                            fp[px * 4 + 0] = b; fp[px * 4 + 1] = g;
                            fp[px * 4 + 2] = r; fp[px * 4 + 3] = a;
                        }
                    }
                }
                else if (pfInfo.pixelType == PIXEL_TYPE_F16)
                {
                    // half4 layout (IEEE 754 float16)
                    uint16_t* fp = reinterpret_cast<uint16_t*>(fillBuf.data());
                    const uint16_t b = Float32ToFloat16(settings.shortageColor.b);
                    const uint16_t g = Float32ToFloat16(settings.shortageColor.g);
                    const uint16_t r = Float32ToFloat16(settings.shortageColor.r);
                    const uint16_t a = Float32ToFloat16(settings.shortageColor.a);
                    for (size_t px = 0; px < sampleImageSize; ++px)
                    {
                        if (arr == 2 /*PIXEL_ARR_ARGB*/)
                        {
                            fp[px * 4 + 0] = a; fp[px * 4 + 1] = r;
                            fp[px * 4 + 2] = g; fp[px * 4 + 3] = b;
                        }
                        else // BGRA
                        {
                            fp[px * 4 + 0] = b; fp[px * 4 + 1] = g;
                            fp[px * 4 + 2] = r; fp[px * 4 + 3] = a;
                        }
                    }
                }
                else if (pfInfo.pixelType == PIXEL_TYPE_U16)
                {
                    // uint16 layout (scaled to [0..32768])
                    uint16_t* fp = reinterpret_cast<uint16_t*>(fillBuf.data());
                    const uint16_t b = static_cast<uint16_t>(frame_echo::Clamp01(settings.shortageColor.b) * 32768.0f);
                    const uint16_t g = static_cast<uint16_t>(frame_echo::Clamp01(settings.shortageColor.g) * 32768.0f);
                    const uint16_t r = static_cast<uint16_t>(frame_echo::Clamp01(settings.shortageColor.r) * 32768.0f);
                    const uint16_t a = static_cast<uint16_t>(frame_echo::Clamp01(settings.shortageColor.a) * 32768.0f);
                    for (size_t px = 0; px < sampleImageSize; ++px)
                    {
                        if (arr == 2 /*PIXEL_ARR_ARGB*/)
                        {
                            fp[px * 4 + 0] = a; fp[px * 4 + 1] = r;
                            fp[px * 4 + 2] = g; fp[px * 4 + 3] = b;
                        }
                        else // BGRA
                        {
                            fp[px * 4 + 0] = b; fp[px * 4 + 1] = g;
                            fp[px * 4 + 2] = r; fp[px * 4 + 3] = a;
                        }
                    }
                }
                else // elementSize == 4, PIXEL_TYPE_U8
                {
                    uint8_t* fp = fillBuf.data();
                    const uint8_t b = static_cast<uint8_t>(frame_echo::Clamp01(settings.shortageColor.b) * 255.0f + 0.5f);
                    const uint8_t g = static_cast<uint8_t>(frame_echo::Clamp01(settings.shortageColor.g) * 255.0f + 0.5f);
                    const uint8_t r = static_cast<uint8_t>(frame_echo::Clamp01(settings.shortageColor.r) * 255.0f + 0.5f);
                    const uint8_t a = static_cast<uint8_t>(frame_echo::Clamp01(settings.shortageColor.a) * 255.0f + 0.5f);
                    for (size_t px = 0; px < sampleImageSize; ++px)
                    {
                        if (arr == 2 /*PIXEL_ARR_ARGB*/)
                        {
                            fp[px * 4 + 0] = a; fp[px * 4 + 1] = r;
                            fp[px * 4 + 2] = g; fp[px * 4 + 3] = b;
                        }
                        else // BGRA
                        {
                            fp[px * 4 + 0] = b; fp[px * 4 + 1] = g;
                            fp[px * 4 + 2] = r; fp[px * 4 + 3] = a;
                        }
                    }
                }

                // clEnqueueWriteBuffer with CL_TRUE is blocking - fillBuf safe to reuse
                for (int s = 0; s < numSamples; ++s)
                {
                    if (s == nb) continue;
                    if (devPtrs[s]) continue;
                    res = clEnqueueWriteBuffer(cq, samplesBuf, CL_TRUE,
                        (size_t)s * sampleByteSize, sampleByteSize, fillBuf.data(),
                        0, nullptr, nullptr);
                    CL_ERR(res);
                }
            }
            // Repeat: no pre-fill needed; devPtrs already resolved by Assemble
        }

        // Copy all valid frames to samples buffer
        if (!err)
        {
            for (int s = 0; s < numSamples; ++s)
            {
                if (!devPtrs[s]) continue;
                res = clEnqueueCopyBuffer(cq, (cl_mem)devPtrs[s], samplesBuf,
                    0, (size_t)s * sampleByteSize, sampleByteSize, 0, nullptr, nullptr);
                CL_ERR(res);
                if (err) break;
            }
        }

        // Weights buffer
        cl_mem wtBuf = nullptr;
        if (!err)
        {
            size_t wtBytes = (size_t)numSamples * sizeof(float);
            wtBuf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, wtBytes, nullptr, &res);
            CL_ERR(res);
            if (!err)
            {
                res = clEnqueueWriteBuffer(cq, wtBuf, CL_TRUE, 0, wtBytes,
                    prep.weights.data(), 0, nullptr, nullptr);
                CL_ERR(res);
            }
        }

        // Launch kernel (FrameEchoBlend_OCL)
        if (!err)
        {
            cl_kernel kernel = sAeOpenCLKernel[extra->input->device_index];
            if (!kernel) err = PF_Err_INTERNAL_STRUCT_DAMAGED;

            if (!err)
            {
                int a = 0;
                int boxLeft = 0, boxTop = 0, boxW = width, boxH = height;
                int blendK = BlendModeToKernelEnum(settings.blendMode);
                int pixelFormatBits = pfInfo.pixelFormatBits;
                int bytePitch = elementPitch * pfInfo.elementSize;

                CL_ERR(clSetKernelArg(kernel, a++, sizeof(cl_mem), &outDevPtr));
                CL_ERR(clSetKernelArg(kernel, a++, sizeof(cl_mem), &samplesBuf));
                CL_ERR(clSetKernelArg(kernel, a++, sizeof(cl_mem), &wtBuf));
                CL_ERR(clSetKernelArg(kernel, a++, sizeof(int), &boxLeft));
                CL_ERR(clSetKernelArg(kernel, a++, sizeof(int), &boxTop));
                CL_ERR(clSetKernelArg(kernel, a++, sizeof(int), &boxW));
                CL_ERR(clSetKernelArg(kernel, a++, sizeof(int), &boxH));
                CL_ERR(clSetKernelArg(kernel, a++, sizeof(int), &bytePitch));
                CL_ERR(clSetKernelArg(kernel, a++, sizeof(int), &height));
                CL_ERR(clSetKernelArg(kernel, a++, sizeof(int), &pixelFormatBits));
                CL_ERR(clSetKernelArg(kernel, a++, sizeof(int), &numSamples));
                CL_ERR(clSetKernelArg(kernel, a++, sizeof(int), &blendK));

                size_t tb[2] = { 16, 16 };
                size_t gd[2] = {
                    ((size_t)width  + 15) / 16 * 16,
                    ((size_t)height + 15) / 16 * 16
                };
                CL_ERR(clEnqueueNDRangeKernel(cq, kernel, 2, nullptr,
                                              gd, tb, 0, nullptr, nullptr));
            }
        }

        // Wait for kernel to finish before releasing buffers
        clFinish(cq);

        // Cleanup
        if (samplesBuf) clReleaseMemObject(samplesBuf);
        if (wtBuf)      clReleaseMemObject(wtBuf);
#endif // GPU_ENABLE_OPENCL
    }

    // ------------------------------------------------------------------
    // CUDA path
    // ------------------------------------------------------------------
    if (!err && extra->input->what_gpu == PF_GPU_Framework_CUDA)
    {
#ifdef GPU_ENABLE_CUDA
        CUdeviceptr outDevCUDA = reinterpret_cast<CUdeviceptr>(outDevPtr);
        CUstream stream = reinterpret_cast<CUstream>(deviceInfo.command_queuePV);
        CUresult cuRes;

        // Allocate consolidated samples buffer
        CUdeviceptr samplesDevPtr = 0;
        cuRes = cuMemAlloc(&samplesDevPtr, totalBytes);
        if (cuRes != CUDA_SUCCESS) err = PF_Err_INTERNAL_STRUCT_DAMAGED;

        // Pre-fill with shortage pattern (BlankTransparent / SolidColor)
        // fillBuf declared outside SolidColor block so it lives until
        // cuStreamSynchronize below (async copies must outlive submissions).
        std::vector<uint8_t> fillBuf;
        if (!err)
        {
            const auto& shortage = settings.frameShortageBehavior;
            if (shortage == frame_echo::FrameShortageBehavior::BlankTransparent)
            {
                for (int s = 0; s < numSamples; ++s)
                {
                    if (s == nb || devPtrs[s]) continue;
                    cuRes = cuMemsetD8(samplesDevPtr + (size_t)s * sampleByteSize, 0, sampleByteSize);
                    if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); return PF_Err_INTERNAL_STRUCT_DAMAGED; }
                }
            }
            else if (shortage == frame_echo::FrameShortageBehavior::SolidColor)
            {
                // Build fill buffer in the native pixel layout
                const int arr = pfInfo.pixelFormatBits & 0x1F;
                fillBuf.resize(sampleByteSize);

                if (pfInfo.elementSize == 16)
                {
                    // float4 layout
                    float* fp = reinterpret_cast<float*>(fillBuf.data());
                    const float b = settings.shortageColor.b;
                    const float g = settings.shortageColor.g;
                    const float r = settings.shortageColor.r;
                    const float a = settings.shortageColor.a;
                    for (size_t px = 0; px < sampleImageSize; ++px)
                    {
                        if (arr == 2 /*PIXEL_ARR_ARGB*/)
                        {
                            fp[px * 4 + 0] = a; fp[px * 4 + 1] = r;
                            fp[px * 4 + 2] = g; fp[px * 4 + 3] = b;
                        }
                        else // BGRA
                        {
                            fp[px * 4 + 0] = b; fp[px * 4 + 1] = g;
                            fp[px * 4 + 2] = r; fp[px * 4 + 3] = a;
                        }
                    }
                }
                else if (pfInfo.pixelType == PIXEL_TYPE_F16)
                {
                    // half4 layout (IEEE 754 float16)
                    uint16_t* fp = reinterpret_cast<uint16_t*>(fillBuf.data());
                    const uint16_t b = Float32ToFloat16(settings.shortageColor.b);
                    const uint16_t g = Float32ToFloat16(settings.shortageColor.g);
                    const uint16_t r = Float32ToFloat16(settings.shortageColor.r);
                    const uint16_t a = Float32ToFloat16(settings.shortageColor.a);
                    for (size_t px = 0; px < sampleImageSize; ++px)
                    {
                        if (arr == 2 /*PIXEL_ARR_ARGB*/)
                        {
                            fp[px * 4 + 0] = a; fp[px * 4 + 1] = r;
                            fp[px * 4 + 2] = g; fp[px * 4 + 3] = b;
                        }
                        else // BGRA
                        {
                            fp[px * 4 + 0] = b; fp[px * 4 + 1] = g;
                            fp[px * 4 + 2] = r; fp[px * 4 + 3] = a;
                        }
                    }
                }
                else if (pfInfo.pixelType == PIXEL_TYPE_U16)
                {
                    // uint16 layout (scaled to [0..32768])
                    uint16_t* fp = reinterpret_cast<uint16_t*>(fillBuf.data());
                    const uint16_t b = static_cast<uint16_t>(frame_echo::Clamp01(settings.shortageColor.b) * 32768.0f);
                    const uint16_t g = static_cast<uint16_t>(frame_echo::Clamp01(settings.shortageColor.g) * 32768.0f);
                    const uint16_t r = static_cast<uint16_t>(frame_echo::Clamp01(settings.shortageColor.r) * 32768.0f);
                    const uint16_t a = static_cast<uint16_t>(frame_echo::Clamp01(settings.shortageColor.a) * 32768.0f);
                    for (size_t px = 0; px < sampleImageSize; ++px)
                    {
                        if (arr == 2 /*PIXEL_ARR_ARGB*/)
                        {
                            fp[px * 4 + 0] = a; fp[px * 4 + 1] = r;
                            fp[px * 4 + 2] = g; fp[px * 4 + 3] = b;
                        }
                        else // BGRA
                        {
                            fp[px * 4 + 0] = b; fp[px * 4 + 1] = g;
                            fp[px * 4 + 2] = r; fp[px * 4 + 3] = a;
                        }
                    }
                }
                else // elementSize == 4, PIXEL_TYPE_U8
                {
                    uint8_t* fp = fillBuf.data();
                    const uint8_t b = static_cast<uint8_t>(frame_echo::Clamp01(settings.shortageColor.b) * 255.0f + 0.5f);
                    const uint8_t g = static_cast<uint8_t>(frame_echo::Clamp01(settings.shortageColor.g) * 255.0f + 0.5f);
                    const uint8_t r = static_cast<uint8_t>(frame_echo::Clamp01(settings.shortageColor.r) * 255.0f + 0.5f);
                    const uint8_t a = static_cast<uint8_t>(frame_echo::Clamp01(settings.shortageColor.a) * 255.0f + 0.5f);
                    for (size_t px = 0; px < sampleImageSize; ++px)
                    {
                        if (arr == 2 /*PIXEL_ARR_ARGB*/)
                        {
                            fp[px * 4 + 0] = a; fp[px * 4 + 1] = r;
                            fp[px * 4 + 2] = g; fp[px * 4 + 3] = b;
                        }
                        else // BGRA
                        {
                            fp[px * 4 + 0] = b; fp[px * 4 + 1] = g;
                            fp[px * 4 + 2] = r; fp[px * 4 + 3] = a;
                        }
                    }
                }

                // fillBuf lives until after cuStreamSynchronize below - async copies safe
                for (int s = 0; s < numSamples; ++s)
                {
                    if (s == nb || devPtrs[s]) continue;
                    cuRes = cuMemcpyHtoDAsync(samplesDevPtr + (size_t)s * sampleByteSize,
                        fillBuf.data(), sampleByteSize, stream);
                    if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); return PF_Err_INTERNAL_STRUCT_DAMAGED; }
                }
            }
            // Repeat: no pre-fill needed; devPtrs already resolved by Assemble
        }

        // Copy all valid frames
        if (!err)
        {
            for (int s = 0; s < numSamples; ++s)
            {
                if (!devPtrs[s]) continue;
                cuRes = cuMemcpyDtoDAsync(
                    samplesDevPtr + (size_t)s * sampleByteSize,
                    reinterpret_cast<CUdeviceptr>(devPtrs[s]), sampleByteSize, stream);
                if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); return PF_Err_INTERNAL_STRUCT_DAMAGED; }
            }
        }

        // Weights buffer
        CUdeviceptr wtDevPtr = 0;
        if (!err)
        {
            size_t wtBytes = (size_t)numSamples * sizeof(float);
            cuRes = cuMemAlloc(&wtDevPtr, wtBytes);
            if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); return PF_Err_INTERNAL_STRUCT_DAMAGED; }
            cuRes = cuMemcpyHtoDAsync(wtDevPtr, prep.weights.data(), wtBytes, stream);
            if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); cuMemFree(wtDevPtr); return PF_Err_INTERNAL_STRUCT_DAMAGED; }
        }

        // Launch kernel (CUDA)
        if (!err)
        {
            int bytePitch = elementPitch * pfInfo.elementSize;
            int pixelFormatBits = pfInfo.pixelFormatBits;
            int boxLeft = 0, boxTop = 0, boxW = width, boxH = height;

            FrameEchoBlend_CUDA_Launch(
                outDevCUDA, samplesDevPtr, wtDevPtr,
                boxLeft, boxTop, boxW, boxH,
                bytePitch, height, pixelFormatBits,
                numSamples, BlendModeToKernelEnum(settings.blendMode),
                stream);
        }

        // Wait for kernel to finish before freeing buffers
        cuStreamSynchronize(stream);

        // Cleanup
        if (samplesDevPtr) cuMemFree(samplesDevPtr);
        if (wtDevPtr)      cuMemFree(wtDevPtr);
#endif // GPU_ENABLE_CUDA
    }

    // ---- Checkin all frames (shared) ----
    AeSmartRenderCheckin(in_data, extra, prep);

    return err;
}
