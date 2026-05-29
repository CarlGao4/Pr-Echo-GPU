//=============================================================================
// FrameEchoPrGpu.cpp - Premiere Pro GPU Filter implementation
//
// Handles Init/Render/Shutdown for OpenCL, DirectX, and CUDA compute backends.
// The Render function consolidates the current frame (from outFrame) and
// all dependency frames (from inFrames[]) into a single GPU buffer, then
// launches the FrameEchoBlend kernel.
//
// CUDA: statically linked via CMake enable_language(CUDA).  The .cu kernel
// is compiled directly into the .aex - no cuModuleLoad needed.
//
// DirectX: Premiere Pro does not currently support DirectX GPU filters.
// The DX path is stubbed out for future use.  Enable via CMake option
// GPU_ENABLE_DIRECTX=ON when Pr adds DX support.
//=============================================================================

#include "FrameEchoCore.h"
#include "FrameEchoParams.h"

#include "PrGPUFilterModule.h"

#ifdef GPU_ENABLE_OPENCL
#include "FrameEchoBlend.cl.h"
#include <CL/cl.h>
#endif

#ifdef GPU_ENABLE_DIRECTX
#include "DirectXUtils.h"
#endif

#ifdef GPU_ENABLE_CUDA
#include <cuda.h>

// Launch wrapper from FrameEchoBlend.cu (compiled by nvcc, linked statically)
extern "C" void FrameEchoBlend_CUDA_Launch(
    CUdeviceptr ioImage, CUdeviceptr inSamples, CUdeviceptr inOpacities,
    int inBoxLeft, int inBoxTop, int inBoxWidth, int inBoxHeight,
    int inBytePitch, int inHeight, int inPixelFormat,
    int inNumSamples, int inBlendMode, CUstream stream);

// Native half4 compute launch wrapper (CC >= 6.0)
extern "C" void FrameEchoBlend_CUDA_Half_Launch(
    CUdeviceptr ioImage, CUdeviceptr inSamples, CUdeviceptr inOpacities,
    int inBoxLeft, int inBoxTop, int inBoxWidth, int inBoxHeight,
    int inHalf4Pitch, int inHeight,
    int inNumSamples, int inBlendMode, CUstream stream);
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

// ============================================================================
// Host-side float32 -> float16 (IEEE 754) conversion for SolidColor fills
// ============================================================================
static inline uint16_t Float32ToFloat16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x007FFFFF;

    if (exp <= 0)
    {
        // Underflow / zero -> flush to zero
        return (uint16_t)sign;
    }
    if (exp >= 31)
    {
        // Infinity or NaN
        return (uint16_t)(sign | 0x7C00 | (mant ? 0x0200 : 0));
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

// ============================================================================
// Pixel format bitfield constants (matching CUDA/OpenCL kernel definitions)
// ============================================================================
#define PIXEL_ARR_BGRA  1
#define PIXEL_TYPE_F32  (2 << 5)
#define PIXEL_TYPE_F16  (3 << 5)

// ============================================================================
// Blend mode -> kernel enum
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
// Cached per-device resources
// ============================================================================
enum { kMaxDevices = 12 };

#ifdef GPU_ENABLE_OPENCL
// OpenCL
static cl_kernel sOpenCLKernelCache[kMaxDevices] = {};
#endif

// DirectX
#ifdef GPU_ENABLE_DIRECTX
static std::vector<DXContextPtr>   sDXContextCache;
static std::vector<ShaderObjectPtr> sShaderObjectCache;
#endif

// ============================================================================
namespace frame_echo
{

class FrameEchoGPU : public PrGPUFilterBase
{
public:
    FrameEchoGPU() :
#ifdef GPU_ENABLE_OPENCL
        mCLKernel(nullptr)
#endif
    {}
    ~FrameEchoGPU() override = default;

    // -------- Static interface --------
    static csSDK_int32 PluginCount() { return 1; }

    static PrSDKString MatchName(piSuitesPtr, csSDK_int32)
    {
        return PrSDKString();
    }

    // -------- Initialize --------
    prSuiteError Initialize(PrGPUFilterInstance* ioInstanceData) override
    {
        PrGPUFilterBase::Initialize(ioInstanceData);
        if (mDeviceIndex >= kMaxDevices) return suiteError_Fail;

        DetectFloat16Support();
#ifdef GPU_ENABLE_OPENCL
        if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_OpenCL)
        {
            prSuiteError err = InitOpenCL();
            return err;
        }
#endif
#ifdef GPU_ENABLE_DIRECTX
        else if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_DirectX)
        {
            return InitDirectX();
        }
#endif
#ifdef GPU_ENABLE_CUDA
        else if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA)
        {
            return InitCUDA();
        }
#endif
        return suiteError_Fail;
    }

private:
    bool mSupportsFloat16_OpenCL = false;
    bool mSupportsFloat16_CUDA = false;
    bool mSupportsFloat16_DirectX = false;

    // Detect whether the GPU device supports native float16 compute.
    void DetectFloat16Support()
    {
        if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_OpenCL)
        {
#ifdef GPU_ENABLE_OPENCL
            cl_device_id device = (cl_device_id)mDeviceInfo.outDeviceHandle;
            // Check for cl_khr_fp16 extension via extension string query
            size_t extSize = 0;
            cl_int res = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &extSize);
            if (res != CL_SUCCESS || extSize == 0) return;

            std::string extensions(extSize, '\0');
            res = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, extSize, &extensions[0], nullptr);
            if (res != CL_SUCCESS) return;

            // Search for cl_khr_fp16 in the semicolon-separated extension list
            mSupportsFloat16_OpenCL = extensions.find("cl_khr_fp16") != std::string::npos;
#else
            return;
#endif
        }
#ifdef GPU_ENABLE_CUDA
        else if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA)
        {
            CUdevice cuDevice = (CUdevice)(uintptr_t)mDeviceInfo.outDeviceHandle;
            int ccMajor = 0;
            CUresult res = cuDeviceGetAttribute(&ccMajor,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
            // Native half throughput requires CC >= 6.0 (Pascal+)
            mSupportsFloat16_CUDA = (res == CUDA_SUCCESS && ccMajor >= 6);
        }
#endif
#ifdef GPU_ENABLE_DIRECTX
        else if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_DirectX)
        {
            // DirectX with -enable-16bit-types and Shader Model 6.2+ supports
            // native half (min16float). Assume supported on modern GPUs.
            mSupportsFloat16_DirectX = true;
            return;
        }
#endif
        return;
    }

#ifdef GPU_ENABLE_OPENCL
    prSuiteError InitOpenCL()
    {
        mCLKernel = sOpenCLKernelCache[mDeviceIndex];
        if (!mCLKernel)
        {
            cl_int result = CL_SUCCESS;
            const char* k16fString = mSupportsFloat16_OpenCL ? "#define GF_OPENCL_SUPPORTS_16F 1\n" : "#define GF_OPENCL_SUPPORTS_16F 0\n";
            size_t sizes[] = { strlen(k16fString), kFrameEchoBlend_OpenCLString_Size };
            const char* strings[] = { k16fString, reinterpret_cast<const char*>(kFrameEchoBlend_OpenCLString) };
            cl_context context = (cl_context)mDeviceInfo.outContextHandle;
            cl_device_id device = (cl_device_id)mDeviceInfo.outDeviceHandle;

            cl_program program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
            if (result != CL_SUCCESS) return suiteError_Fail;

            result = clBuildProgram(program, 1, &device,
                "-cl-single-precision-constant -cl-fast-relaxed-math",
                nullptr, nullptr);
            if (result != CL_SUCCESS) return suiteError_Fail;

            mCLKernel = clCreateKernel(program, "FrameEchoBlendKernel", &result);
            if (result != CL_SUCCESS) return suiteError_Fail;

            sOpenCLKernelCache[mDeviceIndex] = mCLKernel;
        }
        return suiteError_NoError;
    }
#endif // GPU_ENABLE_OPENCL

#ifdef GPU_ENABLE_DIRECTX
    prSuiteError InitDirectX()
    {
        if (mDeviceIndex >= static_cast<csSDK_int32>(sDXContextCache.size()))
        {
            sDXContextCache.resize(mDeviceIndex + 1);
            sShaderObjectCache.resize(mDeviceIndex + 1);
        }
        if (!sDXContextCache[mDeviceIndex])
        {
            auto dxCtx = std::make_shared<DXContext>();
            if (!dxCtx->Initialize(
                (ID3D12Device*)mDeviceInfo.outDeviceHandle,
                (ID3D12CommandQueue*)mDeviceInfo.outCommandQueueHandle))
                return suiteError_Fail;

            std::wstring csoPath, sigPath;
            if (!GetShaderPath(L"FrameEchoBlend", csoPath, sigPath))
                return suiteError_Fail;

            auto shaderObj = std::make_shared<ShaderObject>();
            if (!dxCtx->LoadShader(csoPath.c_str(), sigPath.c_str(), shaderObj))
                return suiteError_Fail;

            sDXContextCache[mDeviceIndex] = dxCtx;
            sShaderObjectCache[mDeviceIndex] = shaderObj;
        }
        return suiteError_NoError;
    }
#endif

#ifdef GPU_ENABLE_CUDA
    prSuiteError InitCUDA()
    {
        // Kernel is statically linked into the binary - nothing to load.
        return suiteError_NoError;
    }
#endif

public:
    // -------- GetFrameDependencies --------
    prSuiteError GetFrameDependencies(
        const PrGPUFilterRenderParams* inRenderParams,
        csSDK_int32* ioQueryIndex,
        PrGPUFilterFrameDependency* outFrameRequirements) override
    {
        PrParam bwdP = GetParam(FRAME_ECHO_BACKWARD_FRAME_COUNT, inRenderParams->inClipTime);
        PrParam fwdP = GetParam(FRAME_ECHO_FORWARD_FRAME_COUNT, inRenderParams->inClipTime);

        int bwdCnt = std::max(0, (int)bwdP.mInt32);
        int fwdCnt = std::max(0, (int)fwdP.mInt32);
        int depCnt = bwdCnt + fwdCnt;
        if (depCnt <= 0) return suiteError_InvalidParms;

        int qi = ioQueryIndex ? *ioQueryIndex : 0;
        if (qi < 0 || qi >= depCnt) return suiteError_InvalidParms;

        bool isBwd = qi < bwdCnt;
        int fi = isBwd ? qi : (qi - bwdCnt);
        PrTime offset = (isBwd ? -1 : 1) * (PrTime)(fi + 1) * inRenderParams->inRenderTicksPerFrame;

        outFrameRequirements->outDependencyType = PrGPUDependency_InputFrame;
        outFrameRequirements->outTrackID = 0;
        outFrameRequirements->outSequenceTime = inRenderParams->inSequenceTime + offset;

        if (ioQueryIndex && qi + 1 < depCnt) *ioQueryIndex = qi + 1;
        return suiteError_NoError;
    }

    // -------- Render --------
    prSuiteError Render(
        const PrGPUFilterRenderParams* inRenderParams,
        const PPixHand* inFrames,
        csSDK_size_t inFrameCount,
        PPixHand* outFrame) override
    {
        // ---- Read all params ----
        TemporalSettings s;
        s.backwardFrameCount = std::max(0, (int)GetParam(FRAME_ECHO_BACKWARD_FRAME_COUNT, inRenderParams->inClipTime).mInt32);
        s.forwardFrameCount  = std::max(0, (int)GetParam(FRAME_ECHO_FORWARD_FRAME_COUNT, inRenderParams->inClipTime).mInt32);
        s.blendMode = ToBlendMode((int)GetParam(FRAME_ECHO_BLEND_MODE, inRenderParams->inClipTime).mInt32 + 1);
        s.syncForwardBackward = GetParam(FRAME_ECHO_SYNC_FORWARD_BACKWARD, inRenderParams->inClipTime).mInt32 != 0;
        // GPU popup params are 0-based; add 1 to match FRAME_ECHO_WINDOW_* constants (1-based)
        int rawFwdFunc = (int)GetParam(FRAME_ECHO_FORWARD_WINDOW_FUNCTION, inRenderParams->inClipTime).mInt32;
        int rawBwdFunc = (int)GetParam(FRAME_ECHO_BACKWARD_WINDOW_FUNCTION, inRenderParams->inClipTime).mInt32;
        s.forwardWindowFunction  = ToWindowFunction(rawFwdFunc + 1);
        s.backwardWindowFunction = ToWindowFunction(rawBwdFunc + 1);
        s.forwardWindowFalloffRatio  = (float)GetParam(FRAME_ECHO_FORWARD_WINDOW_FALLOFF_RATIO,  inRenderParams->inClipTime).mFloat64;
        s.backwardWindowFalloffRatio = (float)GetParam(FRAME_ECHO_BACKWARD_WINDOW_FALLOFF_RATIO, inRenderParams->inClipTime).mFloat64;
        s.forwardWindowFalloffMapping  = ((int)GetParam(FRAME_ECHO_FORWARD_WINDOW_FALLOFF_MAPPING,  inRenderParams->inClipTime).mInt32 + 1 == FRAME_ECHO_WINDOW_FALLOFF_INPUT_MAPPED)
            ? WindowFalloffMapping::InputMapped : WindowFalloffMapping::OutputMapped;
        s.backwardWindowFalloffMapping = ((int)GetParam(FRAME_ECHO_BACKWARD_WINDOW_FALLOFF_MAPPING, inRenderParams->inClipTime).mInt32 + 1 == FRAME_ECHO_WINDOW_FALLOFF_INPUT_MAPPED)
            ? WindowFalloffMapping::InputMapped : WindowFalloffMapping::OutputMapped;
        s.forwardMaxOpacity  = (float)GetParam(FRAME_ECHO_FORWARD_MAX_OPACITY,  inRenderParams->inClipTime).mFloat64;
        s.backwardMaxOpacity = (float)GetParam(FRAME_ECHO_BACKWARD_MAX_OPACITY, inRenderParams->inClipTime).mFloat64;

        // ---- Build weights ----
        // When syncForwardBackward is enabled, backward uses forward's window settings (CPU side does the same)
        auto bwdW = BuildDirectionalWeights(s.backwardFrameCount,
            s.syncForwardBackward ? s.forwardWindowFunction  : s.backwardWindowFunction,
            s.syncForwardBackward ? s.forwardWindowFalloffRatio  : s.backwardWindowFalloffRatio,
            s.syncForwardBackward ? s.forwardWindowFalloffMapping : s.backwardWindowFalloffMapping,
            s.syncForwardBackward ? s.forwardMaxOpacity : s.backwardMaxOpacity);
        auto fwdW = BuildDirectionalWeights(s.forwardFrameCount, s.forwardWindowFunction,
                                        s.forwardWindowFalloffRatio, s.forwardWindowFalloffMapping,
                                        s.forwardMaxOpacity);
        int nb = (int)bwdW.size(), nf = (int)fwdW.size();
        int numSamples = 1 + nb + nf; // current + backward + forward

        // ---- Output frame info ----
        PrPixelFormat pixelFormat;
        mPPixSuite->GetPixelFormat(*outFrame, &pixelFormat);
        prRect bounds = {};
        mPPixSuite->GetBounds(*outFrame, &bounds);
        int w = bounds.right - bounds.left, h = bounds.bottom - bounds.top;
        csSDK_int32 rowBytes = 0;
        mPPixSuite->GetRowBytes(*outFrame, &rowBytes);

        // Determine format: float32 (32f) or float16 (16f)
        bool is16f = (pixelFormat == PrPixelFormat_GPU_BGRA_4444_16f);
        bool is32f = (pixelFormat == PrPixelFormat_GPU_BGRA_4444_32f);
        if (!is16f && !is32f)
            return suiteError_Fail;

        // Element pitch: 4 floats (16 B) for 32f, 4 halfs (8 B) for 16f
        int elementSize   = is16f ? 8 : 16;
        int elementPitch  = rowBytes / elementSize; // elements per row
        size_t sampleImgSize = (size_t)elementPitch * (size_t)h; // elements per sample image

        // ---- Read shortage params ----
        {
            int rawShortage = (int)GetParam(FRAME_ECHO_FRAME_SHORTAGE_BEHAVIOR, inRenderParams->inClipTime).mInt32;
            s.frameShortageBehavior = ToFrameShortageBehavior(rawShortage + 1);
            PrParam colorRaw = GetParam(FRAME_ECHO_FRAME_SHORTAGE_COLOR, inRenderParams->inClipTime);

            if (colorRaw.mType == kPrParamType_Int64)
            {
                uint64_t packed = (uint64_t)colorRaw.mInt64;
                s.shortageColor.b = ((packed >> 8)  & 0xFF) / 255.0f;
                s.shortageColor.g = ((packed >> 24) & 0xFF) / 255.0f;
                s.shortageColor.r = ((packed >> 40) & 0xFF) / 255.0f;
                s.shortageColor.a = ((packed >> 56) & 0xFF) / 255.0f;
            }
            else if (colorRaw.mType == kPrParamType_Int32)
            {
                uint32_t packed = (uint32_t)colorRaw.mInt32;
                s.shortageColor.b = ((packed >> 0)  & 0xFF) / 255.0f;
                s.shortageColor.g = ((packed >> 8)  & 0xFF) / 255.0f;
                s.shortageColor.r = ((packed >> 16) & 0xFF) / 255.0f;
                s.shortageColor.a = ((packed >> 24) & 0xFF) / 255.0f;
            }
            else
            {
                s.shortageColor = PixelRGBA{0.0f, 0.0f, 0.0f, 0.0f};
            }
        }

        // ---- Query clip boundary ----
        PrTime clipStart = 0, clipEnd = 0;
        {
            csSDK_int32 vsID = 0;
            prSuiteError serr = mVideoSegmentSuite->AcquireVideoSegmentsID(mTimelineID, &vsID);
            if (PrSuiteErrorSucceeded(serr) && vsID)
            {
                csSDK_int32 segCount = 0;
                mVideoSegmentSuite->GetSegmentCount(vsID, &segCount);
                for (csSDK_int32 segIdx = 0; segIdx < segCount; ++segIdx)
                {
                    PrTime segStart = 0, segEnd = 0, segOffset = 0;
                    prPluginID segHash = {};
                    serr = mVideoSegmentSuite->GetSegmentInfo(vsID, segIdx,
                        &segStart, &segEnd, &segOffset, &segHash);
                    if (PrSuiteErrorSucceeded(serr) &&
                        inRenderParams->inSequenceTime >= segStart &&
                        inRenderParams->inSequenceTime < segEnd)
                    {
                        clipStart = segStart;
                        clipEnd = segEnd;
                        break;
                    }
                }
                mVideoSegmentSuite->ReleaseVideoSegmentsID(vsID);
            }
        }

        // ---- Determine which frames are valid (within clip bounds) ----
        PrTime ticksPerFrame = inRenderParams->inRenderTicksPerFrame;
        PrTime curSeq = inRenderParams->inSequenceTime;

        // validBackward[i]: i=0 is qi=0 (nearest bwd, offset=-1)
        std::vector<bool> validBackward(nb, false);
        for (int i = 0; i < nb; ++i)
        {
            PrTime expected = curSeq - (PrTime)(i + 1) * ticksPerFrame;
            validBackward[i] = (expected >= clipStart);
        }

        // validForward[i]: i=0 is qi=nb (nearest fwd, offset=+1)
        std::vector<bool> validForward(nf, false);
        for (int i = 0; i < nf; ++i)
        {
            PrTime expected = curSeq + (PrTime)(i + 1) * ticksPerFrame;
            validForward[i] = (expected < clipEnd);
        }

        // ---- Build opacities in chronological order ----
        // Chronological slots: [farthest_bwd ... nearest_bwd, current, nearest_fwd ... farthest_fwd]
        std::vector<float> opacities(numSamples);
        opacities[nb] = GetCurrentFrameWeight(s);
        for (int i = 0; i < nb; ++i)
        {
            int slot = i;
            int bwdIdx = nb - 1 - i;  // farthest bwd -> bwdW[nb-1], nearest -> bwdW[0]
            opacities[slot] = bwdW[bwdIdx];
        }
        for (int i = 0; i < nf; ++i)
        {
            int slot = nb + 1 + i;
            opacities[slot] = fwdW[i];  // fwdW[0]=nearest, fwdW[nf-1]=farthest
        }

        // ---- Get output GPU data ----
        void* outData = nullptr;
        mGPUDeviceSuite->GetGPUPPixData(*outFrame, &outData);
        if (!outData) return suiteError_Fail;

#ifdef GPU_ENABLE_OPENCL
        // ---- Build consolidated samples buffer and dispatch ----
        if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_OpenCL)
        {
            return RenderOpenCL_Ordered(outData, inFrames, inFrameCount,
                opacities, numSamples, elementPitch, w, h, sampleImgSize,
                s, nb, nf, validBackward, validForward, is16f);
        }
#endif
#ifdef GPU_ENABLE_DIRECTX
        else if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_DirectX)
        {
            // NOTE: Premiere Pro does not currently support DirectX GPU filters.
            // This path is provided for future preparation only and is untested.
            return RenderDirectX_Ordered(outData, inFrames, inFrameCount,
                opacities, numSamples, elementPitch, w, h, sampleImgSize,
                s, nb, nf, validBackward, validForward, is16f, rowBytes);
        }
#endif
#ifdef GPU_ENABLE_CUDA
        else if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA)
        {
            if (is16f && mSupportsFloat16_CUDA)
            {
                return RenderCUDA_Half_Ordered(outData, inFrames, inFrameCount,
                    opacities, numSamples, elementPitch, w, h, sampleImgSize,
                    s, nb, nf, validBackward, validForward);
            }
            else
            {
                return RenderCUDA_Ordered(outData, inFrames, inFrameCount,
                    opacities, numSamples, elementPitch, rowBytes, w, h, sampleImgSize,
                    s, nb, nf, validBackward, validForward, pixelFormat);
            }
        }
#endif
        return suiteError_Fail;
    }

    // -------- Shutdown --------
    static prSuiteError Shutdown(piSuitesPtr, csSDK_int32 inIndex)
    {
#ifdef GPU_ENABLE_DIRECTX
        if (inIndex < (csSDK_int32)sDXContextCache.size() && sDXContextCache[inIndex])
            sDXContextCache[inIndex].reset();
        if (inIndex < (csSDK_int32)sShaderObjectCache.size() && sShaderObjectCache[inIndex])
            sShaderObjectCache[inIndex].reset();
#endif
        return suiteError_NoError;
    }

private:
    // ---- Build consolidated buffer + launch for each backend ----

#ifdef GPU_ENABLE_OPENCL
    // ---- OpenCL Render (chronological order) ----
    prSuiteError RenderOpenCL_Ordered(
        void* outData,
        const PPixHand* inFrames,
        csSDK_size_t inFrameCount,
        const std::vector<float>& opacities,
        int numSamples,
        int elementPitch, int w, int h,
        size_t sampleImgSize,
        const TemporalSettings& settings,
        int nb, int nf,
        const std::vector<bool>& validBackward,
        const std::vector<bool>& validForward,
        bool is16f)
    {
        cl_context ctx       = (cl_context)mDeviceInfo.outContextHandle;
        cl_command_queue cq  = (cl_command_queue)mDeviceInfo.outCommandQueueHandle;
        cl_int res;
        int elementSize     = is16f ? 8 : 16;
        size_t sampleByteSize = sampleImgSize * elementSize;
        size_t totalBytes   = (size_t)numSamples * sampleByteSize;

        cl_mem samplesBuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, totalBytes, nullptr, &res);
        if (res != CL_SUCCESS) return suiteError_Fail;

        // ---- Pre-fill all non-current slots with shortage pattern ----
        size_t currentOff = (size_t)nb * sampleByteSize;

        if (settings.frameShortageBehavior == FrameShortageBehavior::BlankTransparent)
        {
            // Zero-fill non-current slots
            cl_int zero = 0;
            for (int s = 0; s < numSamples; ++s)
            {
                if (s == nb) continue;
                size_t off = (size_t)s * sampleByteSize;
                res = clEnqueueFillBuffer(cq, samplesBuf, &zero, sizeof(cl_int),
                                          off, sampleByteSize, 0, nullptr, nullptr);
                if (res != CL_SUCCESS) { clReleaseMemObject(samplesBuf); return suiteError_Fail; }
            }
        }
        else if (settings.frameShortageBehavior == FrameShortageBehavior::Repeat)
        {
            // Repeat: skip pre-fill; handled by post-fill after valid frames are placed
        }
        else // SolidColor
        {
            // Build host-side fill buffer (BGRA pixel repeated)
            std::vector<uint8_t> fillBuf(sampleByteSize);
            if (is16f)
            {
                uint16_t* fp = reinterpret_cast<uint16_t*>(fillBuf.data());
                uint16_t b = Float32ToFloat16(settings.shortageColor.b);
                uint16_t g = Float32ToFloat16(settings.shortageColor.g);
                uint16_t r = Float32ToFloat16(settings.shortageColor.r);
                uint16_t a = Float32ToFloat16(settings.shortageColor.a);
                for (size_t px = 0; px < sampleImgSize; ++px)
                {
                    fp[px * 4 + 0] = b;
                    fp[px * 4 + 1] = g;
                    fp[px * 4 + 2] = r;
                    fp[px * 4 + 3] = a;
                }
            }
            else
            {
                float* fp = reinterpret_cast<float*>(fillBuf.data());
                float b = settings.shortageColor.b;
                float g = settings.shortageColor.g;
                float r = settings.shortageColor.r;
                float a = settings.shortageColor.a;
                for (size_t px = 0; px < sampleImgSize; ++px)
                {
                    fp[px * 4 + 0] = b;
                    fp[px * 4 + 1] = g;
                    fp[px * 4 + 2] = r;
                    fp[px * 4 + 3] = a;
                }
            }
            // Write to all non-current slots
            for (int s = 0; s < numSamples; ++s)
            {
                if (s == nb) continue;
                size_t off = (size_t)s * sampleByteSize;
                res = clEnqueueWriteBuffer(cq, samplesBuf, CL_TRUE, off,
                                           sampleByteSize, fillBuf.data(), 0, nullptr, nullptr);
                if (res != CL_SUCCESS) { clReleaseMemObject(samplesBuf); return suiteError_Fail; }
            }
        }

        // ---- Overwrite valid backward frames ----
        for (int slot = 0; slot < nb; ++slot)
        {
            int qi = nb - 1 - slot;
            if (!validBackward[qi]) continue;
            int inIdx = qi + 1;
            void* src = nullptr;
            if (inIdx < (int)inFrameCount && inFrames[inIdx])
                mGPUDeviceSuite->GetGPUPPixData(inFrames[inIdx], &src);
            if (!src) continue;
            // Device-to-device copy: Pr's inFrames buffer -> samplesBuf
            res = clEnqueueCopyBuffer(cq, (cl_mem)src, samplesBuf, 0,
                (size_t)slot * sampleByteSize, sampleByteSize, 0, nullptr, nullptr);
            if (res != CL_SUCCESS) { clReleaseMemObject(samplesBuf); return suiteError_Fail; }
        }

        // ---- Copy current frame to slot nb ----
        res = clEnqueueCopyBuffer(cq, (cl_mem)outData, samplesBuf, 0,
            currentOff, sampleByteSize, 0, nullptr, nullptr);
        if (res != CL_SUCCESS) { clReleaseMemObject(samplesBuf); return suiteError_Fail; }

        // ---- Overwrite valid forward frames ----
        for (int slot = nb + 1; slot < numSamples; ++slot)
        {
            int fwdIdx = slot - nb - 1;
            if (!validForward[fwdIdx]) continue;
            int inIdx = nb + fwdIdx + 1;
            void* src = nullptr;
            if (inIdx < (int)inFrameCount && inFrames[inIdx])
                mGPUDeviceSuite->GetGPUPPixData(inFrames[inIdx], &src);
            if (!src) continue;
            // Device-to-device copy: Pr's inFrames buffer -> samplesBuf
            res = clEnqueueCopyBuffer(cq, (cl_mem)src, samplesBuf, 0,
                (size_t)slot * sampleByteSize, sampleByteSize, 0, nullptr, nullptr);
            if (res != CL_SUCCESS) { clReleaseMemObject(samplesBuf); return suiteError_Fail; }
        }

        // ---- Repeat post-fill: fill missing slots with nearest valid frame ----
        if (settings.frameShortageBehavior == FrameShortageBehavior::Repeat)
        {
            // Backward: scan RIGHT toward current for nearest valid
            for (int slot = 0; slot < nb; ++slot)
            {
                int qi = nb - 1 - slot;
                if (validBackward[qi]) continue;
                for (int scan = slot + 1; scan <= nb; ++scan)
                {
                    bool scanValid = (scan == nb) || (scan < nb && validBackward[nb - 1 - scan]);
                    if (scanValid)
                    {
                        res = clEnqueueCopyBuffer(cq, samplesBuf, samplesBuf,
                            (size_t)scan * sampleByteSize, (size_t)slot * sampleByteSize,
                            sampleByteSize, 0, nullptr, nullptr);
                        if (res != CL_SUCCESS) { clReleaseMemObject(samplesBuf); return suiteError_Fail; }
                        break;
                    }
                }
            }
            // Forward: scan LEFT toward current for nearest valid
            for (int slot = nb + 1; slot < numSamples; ++slot)
            {
                int fwdIdx = slot - nb - 1;
                if (validForward[fwdIdx]) continue;
                for (int scan = slot - 1; scan >= 0; --scan)
                {
                    bool scanValid = (scan == nb) ||
                        (scan < nb && validBackward[nb - 1 - scan]) ||
                        (scan > nb && validForward[scan - nb - 1]);
                    if (scanValid)
                    {
                        res = clEnqueueCopyBuffer(cq, samplesBuf, samplesBuf,
                            (size_t)scan * sampleByteSize, (size_t)slot * sampleByteSize,
                            sampleByteSize, 0, nullptr, nullptr);
                        if (res != CL_SUCCESS) { clReleaseMemObject(samplesBuf); return suiteError_Fail; }
                        break;
                    }
                }
            }
        }

        // Opacities buffer
        size_t opBytes = (size_t)numSamples * sizeof(float);
        cl_mem opBuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, opBytes, nullptr, &res);
        if (res != CL_SUCCESS) { clReleaseMemObject(samplesBuf); return suiteError_Fail; }
        clEnqueueWriteBuffer(cq, opBuf, CL_TRUE, 0, opBytes, opacities.data(), 0, nullptr, nullptr);

        // Set kernel args
        cl_mem outMem = (cl_mem)outData;
        int a = 0;
        int boxLeft = 0, boxTop = 0, boxW = w, boxH = h;
        int blendKernel = BlendModeToKernelEnum(settings.blendMode);
        clSetKernelArg(mCLKernel, a++, sizeof(cl_mem), &outMem);
        clSetKernelArg(mCLKernel, a++, sizeof(cl_mem), &samplesBuf);
        clSetKernelArg(mCLKernel, a++, sizeof(cl_mem), &opBuf);
        clSetKernelArg(mCLKernel, a++, sizeof(int), &boxLeft);
        clSetKernelArg(mCLKernel, a++, sizeof(int), &boxTop);
        clSetKernelArg(mCLKernel, a++, sizeof(int), &boxW);
        clSetKernelArg(mCLKernel, a++, sizeof(int), &boxH);
        clSetKernelArg(mCLKernel, a++, sizeof(int), &elementPitch);
        clSetKernelArg(mCLKernel, a++, sizeof(int), &h);
        clSetKernelArg(mCLKernel, a++, sizeof(int), &numSamples);
        clSetKernelArg(mCLKernel, a++, sizeof(int), &blendKernel);
        int is16fInt = is16f ? 1 : 0;
        clSetKernelArg(mCLKernel, a++, sizeof(int), &is16fInt);
        clFinish(cq);

        size_t tb[2] = { 16, 16 };
        size_t gd[2] = { ((size_t)w + 15) / 16 * 16, ((size_t)h + 15) / 16 * 16 };
        res = clEnqueueNDRangeKernel(cq, mCLKernel, 2, nullptr, gd, tb, 0, nullptr, nullptr);
        clFinish(cq);

        clReleaseMemObject(samplesBuf);
        clReleaseMemObject(opBuf);
        return (res == CL_SUCCESS) ? suiteError_NoError : suiteError_Fail;
    }
#endif // GPU_ENABLE_OPENCL

#ifdef GPU_ENABLE_CUDA
    // ---- CUDA Render (float32 path, chronological order) ----
    prSuiteError RenderCUDA_Ordered(
        void* outData,
        const PPixHand* inFrames,
        csSDK_size_t inFrameCount,
        const std::vector<float>& opacities,
        int numSamples,
        int elementPitch, int rowBytes, int w, int h,
        size_t sampleImgSize,
        const TemporalSettings& settings,
        int nb, int nf,
        const std::vector<bool>& validBackward,
        const std::vector<bool>& validForward,
        PrPixelFormat pixelFormat)
    {
        CUdeviceptr outDevPtr = reinterpret_cast<CUdeviceptr>(outData);
        CUstream stream       = reinterpret_cast<CUstream>(mDeviceInfo.outCommandQueueHandle);
        CUresult cuRes;

        int elementSize = (pixelFormat == PrPixelFormat_GPU_BGRA_4444_16f) ? 8 : 16;
        size_t sampleByteSize = sampleImgSize * elementSize;
        size_t totalBytes     = (size_t)numSamples * sampleByteSize;

        int pfArr = PIXEL_ARR_BGRA;
        int pfType = (pixelFormat == PrPixelFormat_GPU_BGRA_4444_16f) ? PIXEL_TYPE_F16 : PIXEL_TYPE_F32;
        int pixelFormatBits = pfArr | pfType;

        CUdeviceptr samplesDevPtr;
        cuRes = cuMemAlloc(&samplesDevPtr, totalBytes);
        if (cuRes != CUDA_SUCCESS) return suiteError_Fail;

        // Pre-fill non-current slots with shortage pattern
        {
            size_t curOff = (size_t)nb * sampleByteSize;
            if (settings.frameShortageBehavior == FrameShortageBehavior::BlankTransparent)
            {
                for (int s = 0; s < numSamples; ++s)
                {
                    if (s == nb) continue;
                    cuRes = cuMemsetD8(samplesDevPtr + (size_t)s * sampleByteSize, 0, sampleByteSize);
                    if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); return suiteError_Fail; }
                }
            }
            else if (settings.frameShortageBehavior == FrameShortageBehavior::Repeat)
            {
                // Repeat: skip pre-fill; handled by post-fill after valid frames are placed
            }
            else // SolidColor
            {
                std::vector<uint8_t> fillBuf(sampleByteSize);
                if (elementSize == 8)
                {
                    uint16_t* fp = reinterpret_cast<uint16_t*>(fillBuf.data());
                    uint16_t b = Float32ToFloat16(settings.shortageColor.b);
                    uint16_t g = Float32ToFloat16(settings.shortageColor.g);
                    uint16_t r = Float32ToFloat16(settings.shortageColor.r);
                    uint16_t a = Float32ToFloat16(settings.shortageColor.a);
                    for (size_t px = 0; px < sampleImgSize; ++px)
                    {
                        fp[px * 4 + 0] = b; fp[px * 4 + 1] = g;
                        fp[px * 4 + 2] = r; fp[px * 4 + 3] = a;
                    }
                }
                else
                {
                    float* fp = reinterpret_cast<float*>(fillBuf.data());
                    float b = settings.shortageColor.b;
                    float g = settings.shortageColor.g;
                    float r = settings.shortageColor.r;
                    float a = settings.shortageColor.a;
                    for (size_t px = 0; px < sampleImgSize; ++px)
                    {
                        fp[px * 4 + 0] = b; fp[px * 4 + 1] = g;
                        fp[px * 4 + 2] = r; fp[px * 4 + 3] = a;
                    }
                }
                for (int s = 0; s < numSamples; ++s)
                {
                    if (s == nb) continue;
                    cuRes = cuMemcpyHtoDAsync(samplesDevPtr + (size_t)s * sampleByteSize,
                        fillBuf.data(), sampleByteSize, stream);
                    if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); return suiteError_Fail; }
                }
            }
        }

        // ---- Overwrite valid backward + forward + current ----
        {
            size_t curOff = (size_t)nb * sampleByteSize;
            for (int slot = 0; slot < nb; ++slot)
            {
                int qi = nb - 1 - slot;
                if (!validBackward[qi]) continue;
                int inIdx = qi + 1;
                void* src = nullptr;
                if (inIdx < (int)inFrameCount && inFrames[inIdx])
                    mGPUDeviceSuite->GetGPUPPixData(inFrames[inIdx], &src);
                if (!src) continue;
                cuRes = cuMemcpyDtoDAsync(samplesDevPtr + (size_t)slot * sampleByteSize,
                    reinterpret_cast<CUdeviceptr>(src), sampleByteSize, stream);
                if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); return suiteError_Fail; }
            }
            cuRes = cuMemcpyDtoDAsync(samplesDevPtr + curOff, outDevPtr, sampleByteSize, stream);
            if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); return suiteError_Fail; }
            for (int slot = nb + 1; slot < numSamples; ++slot)
            {
                int fwdIdx = slot - nb - 1;
                if (!validForward[fwdIdx]) continue;
                int inIdx = nb + fwdIdx + 1;
                void* src = nullptr;
                if (inIdx < (int)inFrameCount && inFrames[inIdx])
                    mGPUDeviceSuite->GetGPUPPixData(inFrames[inIdx], &src);
                if (!src) continue;
                cuRes = cuMemcpyDtoDAsync(samplesDevPtr + (size_t)slot * sampleByteSize,
                    reinterpret_cast<CUdeviceptr>(src), sampleByteSize, stream);
                if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); return suiteError_Fail; }
            }
        }

        // ---- Repeat post-fill: fill missing slots with nearest valid frame ----
        if (settings.frameShortageBehavior == FrameShortageBehavior::Repeat)
        {
            // Backward: scan RIGHT toward current for nearest valid
            for (int slot = 0; slot < nb; ++slot)
            {
                int qi = nb - 1 - slot;
                if (validBackward[qi]) continue;
                for (int scan = slot + 1; scan <= nb; ++scan)
                {
                    bool scanValid = (scan == nb) || (scan < nb && validBackward[nb - 1 - scan]);
                    if (scanValid)
                    {
                        cuRes = cuMemcpyDtoDAsync(samplesDevPtr + (size_t)slot * sampleByteSize,
                            samplesDevPtr + (size_t)scan * sampleByteSize, sampleByteSize, stream);
                        if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); return suiteError_Fail; }
                        break;
                    }
                }
            }
            // Forward: scan LEFT toward current for nearest valid
            for (int slot = nb + 1; slot < numSamples; ++slot)
            {
                int fwdIdx = slot - nb - 1;
                if (validForward[fwdIdx]) continue;
                for (int scan = slot - 1; scan >= 0; --scan)
                {
                    bool scanValid = (scan == nb) ||
                        (scan < nb && validBackward[nb - 1 - scan]) ||
                        (scan > nb && validForward[scan - nb - 1]);
                    if (scanValid)
                    {
                        cuRes = cuMemcpyDtoDAsync(samplesDevPtr + (size_t)slot * sampleByteSize,
                            samplesDevPtr + (size_t)scan * sampleByteSize, sampleByteSize, stream);
                        if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); return suiteError_Fail; }
                        break;
                    }
                }
            }
        }

        // Opacities buffer
        size_t opBytes = (size_t)numSamples * sizeof(float);
        CUdeviceptr opDevPtr;
        cuRes = cuMemAlloc(&opDevPtr, opBytes);
        if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); return suiteError_Fail; }
        cuRes = cuMemcpyHtoDAsync(opDevPtr, opacities.data(), opBytes, stream);
        if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); cuMemFree(opDevPtr); return suiteError_Fail; }

        // Launch
        int boxLeft = 0, boxTop = 0, boxW = w, boxH = h;
        int bytePitch = elementPitch * elementSize;

        FrameEchoBlend_CUDA_Launch(
            outDevPtr,           samplesDevPtr,       opDevPtr,
            boxLeft,             boxTop,              boxW,
            boxH,                bytePitch,           (int)h,
            pixelFormatBits,     numSamples,          BlendModeToKernelEnum(settings.blendMode),
            stream);

        cuMemFree(samplesDevPtr);
        cuMemFree(opDevPtr);
        return suiteError_NoError;
    }

    // ---- CUDA Render Half (native half4 compute, chronological order) ----
    prSuiteError RenderCUDA_Half_Ordered(
        void* outData,
        const PPixHand* inFrames,
        csSDK_size_t inFrameCount,
        const std::vector<float>& opacities,
        int numSamples,
        int half4Pitch, int w, int h,
        size_t sampleImgSize,
        const TemporalSettings& settings,
        int nb, int nf,
        const std::vector<bool>& validBackward,
        const std::vector<bool>& validForward)
    {
        CUdeviceptr outDevPtr = reinterpret_cast<CUdeviceptr>(outData);
        CUstream stream       = reinterpret_cast<CUstream>(mDeviceInfo.outCommandQueueHandle);
        CUresult cuRes;

        size_t halfSize = sizeof(std::uint16_t) * 4; // half4 = 8 bytes
        size_t sampleByteSize = sampleImgSize * halfSize;
        size_t totalBytes     = (size_t)numSamples * sampleByteSize;

        CUdeviceptr samplesDevPtr;
        cuRes = cuMemAlloc(&samplesDevPtr, totalBytes);
        if (cuRes != CUDA_SUCCESS) return suiteError_Fail;

        // ---- Pre-fill non-current slots with shortage pattern ----
        {
            if (settings.frameShortageBehavior == FrameShortageBehavior::BlankTransparent)
            {
                for (int s = 0; s < numSamples; ++s)
                {
                    if (s == nb) continue;
                    cuRes = cuMemsetD8(samplesDevPtr + (size_t)s * sampleByteSize, 0, sampleByteSize);
                    if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); return suiteError_Fail; }
                }
            }
            else if (settings.frameShortageBehavior == FrameShortageBehavior::Repeat)
            {
                // Repeat: skip pre-fill; handled by post-fill after valid frames are placed
            }
            else // SolidColor
            {
                std::vector<uint8_t> fillBuf(sampleByteSize);
                uint16_t* fp = reinterpret_cast<uint16_t*>(fillBuf.data());
                uint16_t b = Float32ToFloat16(settings.shortageColor.b);
                uint16_t g = Float32ToFloat16(settings.shortageColor.g);
                uint16_t r = Float32ToFloat16(settings.shortageColor.r);
                uint16_t a = Float32ToFloat16(settings.shortageColor.a);
                for (size_t px = 0; px < sampleImgSize; ++px)
                {
                    fp[px * 4 + 0] = b; fp[px * 4 + 1] = g;
                    fp[px * 4 + 2] = r; fp[px * 4 + 3] = a;
                }
                for (int s = 0; s < numSamples; ++s)
                {
                    if (s == nb) continue;
                    cuRes = cuMemcpyHtoDAsync(samplesDevPtr + (size_t)s * sampleByteSize,
                        fillBuf.data(), sampleByteSize, stream);
                    if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); return suiteError_Fail; }
                }
            }
        }

        // ---- Overwrite valid backward + forward + current ----
        {
            size_t curOff = (size_t)nb * sampleByteSize;
            for (int slot = 0; slot < nb; ++slot)
            {
                int qi = nb - 1 - slot;
                if (!validBackward[qi]) continue;
                int inIdx = qi + 1;
                void* src = nullptr;
                if (inIdx < (int)inFrameCount && inFrames[inIdx])
                    mGPUDeviceSuite->GetGPUPPixData(inFrames[inIdx], &src);
                if (!src) continue;
                cuRes = cuMemcpyDtoDAsync(samplesDevPtr + (size_t)slot * sampleByteSize,
                    reinterpret_cast<CUdeviceptr>(src), sampleByteSize, stream);
                if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); return suiteError_Fail; }
            }
            cuRes = cuMemcpyDtoDAsync(samplesDevPtr + curOff, outDevPtr, sampleByteSize, stream);
            if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); return suiteError_Fail; }
            for (int slot = nb + 1; slot < numSamples; ++slot)
            {
                int fwdIdx = slot - nb - 1;
                if (!validForward[fwdIdx]) continue;
                int inIdx = nb + fwdIdx + 1;
                void* src = nullptr;
                if (inIdx < (int)inFrameCount && inFrames[inIdx])
                    mGPUDeviceSuite->GetGPUPPixData(inFrames[inIdx], &src);
                if (!src) continue;
                cuRes = cuMemcpyDtoDAsync(samplesDevPtr + (size_t)slot * sampleByteSize,
                    reinterpret_cast<CUdeviceptr>(src), sampleByteSize, stream);
                if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); return suiteError_Fail; }
            }
        }

        // ---- Repeat post-fill: fill missing slots with nearest valid frame ----
        if (settings.frameShortageBehavior == FrameShortageBehavior::Repeat)
        {
            // Backward: scan RIGHT toward current for nearest valid
            for (int slot = 0; slot < nb; ++slot)
            {
                int qi = nb - 1 - slot;
                if (validBackward[qi]) continue;
                for (int scan = slot + 1; scan <= nb; ++scan)
                {
                    bool scanValid = (scan == nb) || (scan < nb && validBackward[nb - 1 - scan]);
                    if (scanValid)
                    {
                        cuRes = cuMemcpyDtoDAsync(samplesDevPtr + (size_t)slot * sampleByteSize,
                            samplesDevPtr + (size_t)scan * sampleByteSize, sampleByteSize, stream);
                        if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); return suiteError_Fail; }
                        break;
                    }
                }
            }
            // Forward: scan LEFT toward current for nearest valid
            for (int slot = nb + 1; slot < numSamples; ++slot)
            {
                int fwdIdx = slot - nb - 1;
                if (validForward[fwdIdx]) continue;
                for (int scan = slot - 1; scan >= 0; --scan)
                {
                    bool scanValid = (scan == nb) ||
                        (scan < nb && validBackward[nb - 1 - scan]) ||
                        (scan > nb && validForward[scan - nb - 1]);
                    if (scanValid)
                    {
                        cuRes = cuMemcpyDtoDAsync(samplesDevPtr + (size_t)slot * sampleByteSize,
                            samplesDevPtr + (size_t)scan * sampleByteSize, sampleByteSize, stream);
                        if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); return suiteError_Fail; }
                        break;
                    }
                }
            }
        }

        // Opacities buffer
        size_t opBytes = (size_t)numSamples * sizeof(float);
        CUdeviceptr opDevPtr;
        cuRes = cuMemAlloc(&opDevPtr, opBytes);
        if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); return suiteError_Fail; }
        cuRes = cuMemcpyHtoDAsync(opDevPtr, opacities.data(), opBytes, stream);
        if (cuRes != CUDA_SUCCESS) { cuMemFree(samplesDevPtr); cuMemFree(opDevPtr); return suiteError_Fail; }

        // Launch native half4 kernel
        int boxLeft = 0, boxTop = 0, boxW = w, boxH = h;

        FrameEchoBlend_CUDA_Half_Launch(
            outDevPtr,           samplesDevPtr,       opDevPtr,
            boxLeft,             boxTop,              boxW,
            boxH,                half4Pitch,          (int)h,
            numSamples,          BlendModeToKernelEnum(settings.blendMode),
            stream);

        cuMemFree(samplesDevPtr);
        cuMemFree(opDevPtr);
        return suiteError_NoError;
    }
#endif

#ifdef GPU_ENABLE_DIRECTX
    // ---- DirectX Render (chronological order) ----
    // NOTE: Premiere Pro does not currently support DirectX GPU filters.
    // The consolidated sample buffer logic (pre-fill shortage pattern ->
    // overwrite valid frames -> Repeat post-fill) mirrors OpenCL/CUDA paths.
    // When Pr adds DirectX support, enable this block by:
    //   1. Verifying the DXShaderExecution API (SetParamBuffer / SetUAV / SRV)
    //   2. Adding proper D3D12 buffer creation for consolidatedSamples
    //   3. Using D3D12 resource mapping or copy queue for inFrames data transfer
    prSuiteError RenderDirectX_Ordered(
        void* outData,
        const PPixHand* inFrames,
        csSDK_size_t inFrameCount,
        const std::vector<float>& opacities,
        int numSamples,
        int elementPitch, int w, int h,
        size_t sampleImgSize,
        const TemporalSettings& settings,
        int nb, int nf,
        const std::vector<bool>& validBackward,
        const std::vector<bool>& validForward,
        bool is16f,
        int rowBytes)
    {
        // Stub: DirectX not yet supported by Premiere Pro
        return suiteError_NotImplemented;
    }
#endif

#ifdef GPU_ENABLE_OPENCL
    cl_kernel mCLKernel;
#endif
};

} // namespace frame_echo

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<frame_echo::FrameEchoGPU>)
