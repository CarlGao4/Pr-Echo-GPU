//=============================================================================
// FrameEchoSmartRender.cpp - Shared SmartRender preparation for Ae
//
// AeSmartRenderPrepare()  - checkout all frames, build weights
// AeSmartRenderCheckin()  - checkin all checked-out frames
//
// Used by both CPU (SmartRenderCPU in FrameEchoEffect.cpp) and GPU
// (AeSmartRenderGPU in FrameEchoAeGpu.cpp) paths.
//=============================================================================

#include "FrameEchoAeSmartRender.h"
#include "FrameEchoCore.h"
#include "FrameEchoParams.h"

#include "AEConfig.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_Macros.h"
#include "AEFX_SuiteHandlerTemplate.h"
#include "PrSDKAESupport.h"

// ============================================================================
// AeSmartRenderPrepare
//
// 1. Checkout all frames (including current) via checkout_layer_pixels
// 2. Build weights array in slotIndex order
// 3. Identify current frame
// ============================================================================
PF_Err AeSmartRenderPrepare(
    PF_InData*           in_data,
    PF_OutData*          out_data,
    PF_SmartRenderExtra* extra,
    AePreRenderData*     aeData,
    AeSmartRenderPrep&   outPrep)
{
    PF_Err err = PF_Err_NONE;

    if (!aeData) return PF_Err_INTERNAL_STRUCT_DAMAGED;

    const int nb = aeData->backwardCount;
    const int nf = aeData->forwardCount;
    outPrep.numSamples = 1 + nb + nf;

    // Checkout all frames (including current) via checkout_layer_pixels
    // NOTE: must checkout at least one input BEFORE checkout_output
    outPrep.checkedFrames.resize(aeData->frames.size());
    for (std::size_t i = 0; i < aeData->frames.size(); ++i)
    {
        const AeFrameEntry& entry = aeData->frames[i];
        AeCheckedFrame& cf = outPrep.checkedFrames[i];
        cf.slotIndex = entry.slotIndex;
        cf.inRange   = entry.inRange;

        if (!entry.inRange) continue;

        PF_EffectWorld* world = nullptr;
        ERR(extra->cb->checkout_layer_pixels(
            in_data->effect_ref,
            static_cast<A_long>(entry.slotIndex),
            &world));

        cf.world = world;
        cf.valid = (world != nullptr && world->width > 0 && world->height > 0);

        if (err) break;
    }

    // Checkout output world (must be after at least one input checkout)
    ERR(extra->cb->checkout_output(in_data->effect_ref, &outPrep.outputWorld));
    if (!outPrep.outputWorld) return PF_Err_INTERNAL_STRUCT_DAMAGED;
    outPrep.width  = outPrep.outputWorld->width;
    outPrep.height = outPrep.outputWorld->height;

    // Identify current frame
    for (std::size_t i = 0; i < outPrep.checkedFrames.size(); ++i)
    {
        if (outPrep.checkedFrames[i].slotIndex == nb && outPrep.checkedFrames[i].valid)
        {
            outPrep.inputWorld = outPrep.checkedFrames[i].world;
            outPrep.inputLayer = reinterpret_cast<PF_LayerDef*>(outPrep.inputWorld);
            break;
        }
    }
    outPrep.outputLayer = reinterpret_cast<PF_LayerDef*>(outPrep.outputWorld);

    // Build weights in slotIndex order
    const frame_echo::TemporalSettings& settings = aeData->settings;
    const float currentWeight = frame_echo::GetCurrentFrameWeight(settings);
    const std::vector<float> backwardWeights = frame_echo::BuildDirectionalWeights(
        nb,
        settings.syncForwardBackward ? settings.forwardWindowFunction : settings.backwardWindowFunction,
        settings.syncForwardBackward ? settings.forwardWindowFalloffRatio : settings.backwardWindowFalloffRatio,
        settings.syncForwardBackward ? settings.forwardWindowFalloffMapping : settings.backwardWindowFalloffMapping,
        settings.syncForwardBackward ? settings.forwardMaxOpacity : settings.backwardMaxOpacity);
    const std::vector<float> forwardWeights = frame_echo::BuildDirectionalWeights(
        nf, settings.forwardWindowFunction,
        settings.forwardWindowFalloffRatio, settings.forwardWindowFalloffMapping,
        settings.forwardMaxOpacity);

    outPrep.weights.resize(outPrep.numSamples);
    for (int s = 0; s < nb; ++s)
        outPrep.weights[s] = backwardWeights[nb - 1 - s];
    outPrep.weights[nb] = currentWeight;
    for (int s = 0; s < nf; ++s)
        outPrep.weights[nb + 1 + s] = forwardWeights[s];

    if (!err && !outPrep.inputLayer) err = PF_Err_INTERNAL_STRUCT_DAMAGED;

    return err;
}

// ============================================================================
// AeSmartRenderAssemble
//
// Full assembly: prepare + pixel format + dimension validation + repeat.
// Used by both CPU (SmartRenderCPU) and GPU (AeSmartRenderGPU) paths.
// ============================================================================
PF_Err AeSmartRenderAssemble(
    PF_InData*           in_data,
    PF_OutData*          out_data,
    PF_SmartRenderExtra* extra,
    AePreRenderData*     aeData,
    AeSmartRenderPrep&   outPrep)
{
    PF_Err err = PF_Err_NONE;

    // 1. Checkout all frames + build weights
    ERR(AeSmartRenderPrepare(in_data, out_data, extra, aeData, outPrep));
    if (err) return err;

    // 2. Detect pixel format
    AEFX_SuiteScoper<PF_WorldSuite2> worldSuite(
        in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
    ERR(worldSuite->PF_GetPixelFormat(outPrep.inputWorld, &outPrep.pixelFormat));
    if (outPrep.pixelFormat == PF_PixelFormat_INVALID)
        outPrep.pixelFormat = PF_PixelFormat_ARGB32;

    // 3. Dimension validation -> update valid
    for (auto& cf : outPrep.checkedFrames)
    {
        if (!cf.valid || !cf.world) continue;
        if (cf.world->width != outPrep.width ||
            cf.world->height != outPrep.height ||
            cf.world->rowbytes != outPrep.outputWorld->rowbytes)
        {
            cf.valid = false;
        }
    }

    // 4. resolvedSlot: valid frames keep their slot, invalid get -1
    const int numSamples = outPrep.numSamples;
    for (auto& cf : outPrep.checkedFrames)
    {
        cf.resolvedSlot = cf.valid ? cf.slotIndex : -1;
    }

    // 5. Repeat: fill resolvedSlot with nearest valid frame
    if (aeData->settings.frameShortageBehavior ==
        frame_echo::FrameShortageBehavior::Repeat)
    {
        const int nb = aeData->backwardCount;

        // Backward (slots 0..nb-1): scan RIGHT toward current
        for (int s = 0; s < nb; ++s)
        {
            if (outPrep.checkedFrames[s].resolvedSlot >= 0) continue;
            for (int scan = s + 1; scan < numSamples; ++scan)
            {
                if (outPrep.checkedFrames[scan].resolvedSlot >= 0)
                {
                    outPrep.checkedFrames[s].resolvedSlot =
                        outPrep.checkedFrames[scan].resolvedSlot;
                    break;
                }
            }
        }

        // Forward (slots nb+1..numSamples-1): scan LEFT toward current
        for (int s = nb + 1; s < numSamples; ++s)
        {
            if (outPrep.checkedFrames[s].resolvedSlot >= 0) continue;
            for (int scan = s - 1; scan >= 0; --scan)
            {
                if (outPrep.checkedFrames[scan].resolvedSlot >= 0)
                {
                    outPrep.checkedFrames[s].resolvedSlot =
                        outPrep.checkedFrames[scan].resolvedSlot;
                    break;
                }
            }
        }
    }

    return err;
}

// ============================================================================
// AeSmartRenderCheckin
// ============================================================================
void AeSmartRenderCheckin(
    PF_InData*           in_data,
    PF_SmartRenderExtra* extra,
    AeSmartRenderPrep&   prep)
{
    for (const auto& cf : prep.checkedFrames)
    {
        if (cf.valid && cf.world)
        {
            extra->cb->checkin_layer_pixels(
                in_data->effect_ref, static_cast<A_long>(cf.slotIndex));
        }
    }
}
