#include "FrameEchoCore.h"

#include <immintrin.h>
#include <cstdint>

// -----------------------------------------------------------------------
// This file is compiled separately with /arch:AVX2 so all _mm_* intrinsics
// emit VEX-coded instructions.  The compiler is also free to use FMA
// (Fused Multiply-Add) via /arch:AVX2 on supporting CPUs.
//
// All helpers operate on __m128 (128-bit / 4 x float) - i.e. SSE2-level
// width.  The /arch:AVX2 flag here primarily brings VEX encoding (which
// avoids the SSE-AVX state-transition penalty) and enables FMA.
//
// The true 256-bit AVX2 acceleration lives in FrameEchoCore_SIMD_AVX2.cpp.
// -----------------------------------------------------------------------

namespace frame_echo
{

// -----------------------------------------------------------------------
// SSE helper functions for PixelRGBA (4 x float)
// -----------------------------------------------------------------------

/// Load 4 floats from a PixelRGBA (unaligned).
static inline __m128 LoadPixel(const PixelRGBA& p)
{
    return _mm_loadu_ps(&p.r);
}

/// Store 4 floats into a PixelRGBA (unaligned).
static inline void StorePixel(PixelRGBA& p, __m128 v)
{
    _mm_storeu_ps(&p.r, v);
}

/// Clamp each lane to [0, 1].
static inline __m128 Clamp01SIMD(__m128 v)
{
    const __m128 zero = _mm_setzero_ps();
    const __m128 one  = _mm_set1_ps(1.0f);
    return _mm_min_ps(_mm_max_ps(v, zero), one);
}

/// Broadcast a scalar float to all 4 lanes.
static inline __m128 Broadcast(float v)
{
    return _mm_set1_ps(v);
}

/// Premultiply: out.rgb = pixel.rgb * clampedAlpha, out.a = clampedAlpha
/// where clampedAlpha = clamp(pixel.a * opacity, 0, 1)
static inline __m128 PremultiplySIMD(__m128 pixel, float opacity)
{
    const __m128 zero = _mm_setzero_ps();
    const __m128 one  = _mm_set1_ps(1.0f);

    // Broadcast alpha lane to all 4 lanes: [a, a, a, a]
    const __m128 alpha = _mm_shuffle_ps(pixel, pixel, _MM_SHUFFLE(3, 3, 3, 3));
    // preAlpha = a * opacity
    const __m128 preAlpha = _mm_mul_ps(alpha, Broadcast(opacity));
    // clampedAlpha = clamp(preAlpha, 0, 1)
    const __m128 clampedAlpha = _mm_min_ps(_mm_max_ps(preAlpha, zero), one);

    // Replace alpha channel of pixel with 1.0 so RGB are multiplied by clampedAlpha
    // and the alpha channel becomes clampedAlpha * 1.0 = clampedAlpha.
    const __m128 rgb_one = _mm_blend_ps(pixel, one, 0b1000);
    return _mm_mul_ps(rgb_one, clampedAlpha);
}

/// Unpremultiply: out.rgb = pixel.rgb / pixel.a (clamped), out.a = pixel.a
/// Returns zero if pixel.a <= 0.
static inline __m128 UnpremultiplySIMD(__m128 pixel)
{
    // Extract alpha lane to scalar for the zero check and reciprocal
    float alpha_f;
    _mm_store_ss(&alpha_f, _mm_move_ss(_mm_setzero_ps(),
        _mm_shuffle_ps(pixel, pixel, _MM_SHUFFLE(3, 3, 3, 3))));

    if (alpha_f <= 0.0f)
    {
        return _mm_setzero_ps();
    }

    const float invAlpha = 1.0f / alpha_f;
    const __m128 invAlphaV = Broadcast(invAlpha);

    // Scale RGB by invAlpha; preserve original alpha channel
    __m128 result = _mm_mul_ps(pixel, invAlphaV);
    result = _mm_blend_ps(result, pixel, 0b1000);

    return Clamp01SIMD(result);
}

/// Over composite: out = src + dst * (1 - src.a)
static inline __m128 OverSIMD(__m128 dst, __m128 src)
{
    // Extract src.a as scalar for the (1 - a) factor
    float srcAlpha_f;
    _mm_store_ss(&srcAlpha_f, _mm_move_ss(_mm_setzero_ps(),
        _mm_shuffle_ps(src, src, _MM_SHUFFLE(3, 3, 3, 3))));

    const float dstFactor = 1.0f - srcAlpha_f;
    const __m128 dstFactorV = Broadcast(dstFactor);

    // dst * (1 - src.a) + src
    return _mm_add_ps(_mm_mul_ps(dst, dstFactorV), src);
}


// -----------------------------------------------------------------------
// ComposeSamples - SSE2 (128-bit) accelerated dispatch target
// -----------------------------------------------------------------------

PixelRGBA ComposeSamples_SSE2(const std::vector<TemporalSample>& samples, BlendMode blendMode)
{
    if (samples.empty())
    {
        return PixelRGBA{};
    }

    if (blendMode == BlendMode::Add)
    {
        __m128 sum = _mm_setzero_ps();

        for (const TemporalSample& sample : samples)
        {
            const __m128 pixel = LoadPixel(sample.pixel);
            const __m128 premul = PremultiplySIMD(pixel, sample.opacity);
            sum = _mm_add_ps(sum, premul);
        }

        sum = Clamp01SIMD(sum);
        PixelRGBA result;
        StorePixel(result, UnpremultiplySIMD(sum));
        return result;
    }

    if (blendMode == BlendMode::Maximum || blendMode == BlendMode::Minimum)
    {
        __m128 value = PremultiplySIMD(LoadPixel(samples.front().pixel), samples.front().opacity);

        for (std::size_t index = 1; index < samples.size(); ++index)
        {
            const __m128 premul = PremultiplySIMD(LoadPixel(samples[index].pixel), samples[index].opacity);
            if (blendMode == BlendMode::Maximum)
            {
                value = _mm_max_ps(value, premul);
            }
            else
            {
                value = _mm_min_ps(value, premul);
            }
        }

        PixelRGBA result;
        StorePixel(result, UnpremultiplySIMD(value));
        return result;
    }

    // Over blending (BlendNewOnTop / BlendNewOnBottom)
    __m128 composed = _mm_setzero_ps();

    for (const TemporalSample& sample : samples)
    {
        const __m128 premul = PremultiplySIMD(LoadPixel(sample.pixel), sample.opacity);
        if (blendMode == BlendMode::BlendNewOnBottom)
        {
            composed = OverSIMD(composed, premul);
        }
        else
        {
            composed = OverSIMD(premul, composed);
        }
    }

    PixelRGBA result;
    StorePixel(result, UnpremultiplySIMD(composed));
    return result;
}

} // namespace frame_echo
