//=============================================================================
// FrameEchoAeGpu.h - After Effects Smart Render GPU support for FrameEcho
//
// Declares:
//   AePreRenderData  - data passed from PreRender to SmartRender
//   GPUDeviceSetup()  - compile/load GPU kernels
//   GPUDeviceSetdown()- release GPU resources
//   SmartRenderGPU()  - GPU render path (Ae GPU world)
//=============================================================================

#pragma once

#include "AEConfig.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_Macros.h"

#ifdef GPU_ENABLED
#include "AE_EffectGPUSuites.h"
#endif

#include "PrSDKAESupport.h"

#include "FrameEchoCore.h"

#include <vector>

// -----------------------------------------------------------------------
// One frame entry, pre-resolved in PreRender.
// checkoutIdL (used by checkout_layer / checkout_layer_pixels) == slotIndex.
// -----------------------------------------------------------------------
struct AeFrameEntry
{
    PrTime   frameTime;   // absolute time of this frame
    int      slotIndex;   // chronological slot in samples buffer:
                          //   0=farthest_bwd, ..., nb-1=nearest_bwd,
                          //   nb=current, nb+1=nearest_fwd, ..., nb+nf=farthest_fwd
    bool     inRange;     // IsTimeInRange result at PreRender time
};

// -----------------------------------------------------------------------
// Checked-out frame world + metadata (filled by shared prep, consumed by
// CPU render and GPU upload)
// -----------------------------------------------------------------------
struct AeCheckedFrame
{
    PF_EffectWorld* world = nullptr;  // checkout_layer_pixels result
    int             slotIndex = -1;
    bool            inRange = false;
    bool            valid = false;    // world != nullptr && dimensions match output
    int             resolvedSlot = -1; // Repeat: actual source slot after resolution
};

// -----------------------------------------------------------------------
// Data packed in PreRender and consumed by SmartRender (CPU + GPU)
// -----------------------------------------------------------------------
struct AePreRenderData
{
    frame_echo::TemporalSettings settings;

    // Number of backward and forward dependency frames
    int backwardCount = 0;
    int forwardCount  = 0;

    // Pixel format of the input world (set at SmartRender time)
    PF_PixelFormat pixelFormat = PF_PixelFormat_INVALID;

    // Pre-resolved frame list (filled by PreRender), including current frame.
    // checkoutIdL == slotIndex for each entry.
    std::vector<AeFrameEntry> frames;
};

// -----------------------------------------------------------------------
// Shared result from SmartRender prep (used by both CPU and GPU paths)
// -----------------------------------------------------------------------
struct AeSmartRenderPrep
{
    std::vector<AeCheckedFrame>   checkedFrames;  // one per AeFrameEntry
    std::vector<float>            weights;         // numSamples, slotIndex order
    PF_EffectWorld*               inputWorld = nullptr;  // current frame
    PF_EffectWorld*               outputWorld = nullptr;
    PF_LayerDef*                  inputLayer = nullptr;  // cast from inputWorld
    PF_LayerDef*                  outputLayer = nullptr; // cast from outputWorld
    int                           width = 0;
    int                           height = 0;
    int                           numSamples = 0;
    PF_PixelFormat                pixelFormat = PF_PixelFormat_INVALID;
};

// Shared prep: checkout all frames via checkout_layer_pixels, build weights
PF_Err AeSmartRenderPrepare(
    PF_InData*           in_data,
    PF_OutData*          out_data,
    PF_SmartRenderExtra* extra,
    AePreRenderData*     aeData,
    AeSmartRenderPrep&   outPrep);

// Assemble: prepare + pixel format detection + dimension validation + repeat resolution
PF_Err AeSmartRenderAssemble(
    PF_InData*           in_data,
    PF_OutData*          out_data,
    PF_SmartRenderExtra* extra,
    AePreRenderData*     aeData,
    AeSmartRenderPrep&   outPrep);

// Checkin all checked-out frames
void AeSmartRenderCheckin(
    PF_InData*           in_data,
    PF_SmartRenderExtra* extra,
    AeSmartRenderPrep&   prep);

// -----------------------------------------------------------------------
// Ae GPU entry points - called from FrameEchoEffect.cpp EffectMain
// Only available when GPU acceleration is compiled in.
// -----------------------------------------------------------------------

#ifdef GPU_ENABLED

PF_Err AeGPUDeviceSetup(
    PF_InData*      in_data,
    PF_OutData*     out_data,
    PF_GPUDeviceSetupExtra* extra);

PF_Err AeGPUDeviceSetdown(
    PF_InData*      in_data,
    PF_OutData*     out_data,
    PF_GPUDeviceSetdownExtra* extra);

PF_Err AeSmartRenderGPU(
    PF_InData*           in_data,
    PF_OutData*          out_data,
    PF_SmartRenderExtra* extra,
    AePreRenderData*     aeData);

#endif // GPU_ENABLED
