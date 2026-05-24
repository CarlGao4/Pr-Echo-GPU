#include "FrameEchoCore.h"

#include <immintrin.h>
#include <algorithm>

// -----------------------------------------------------------------------
// This file is compiled separately with /arch:AVX2 so all _mm_* / _mm256_*
// intrinsics emit VEX-coded instructions.
//
// All helpers operate on __m256 (256-bit / 8 x float) to process TWO
// PixelRGBA values simultaneously.
// -----------------------------------------------------------------------

namespace frame_echo
{

// =======================================================================
// AVX2 (256-bit) helpers - process TWO pixels (8 x float) at once.
// =======================================================================

/// Load two PixelRGBA values into a __m256 register.
/// Layout: low 128 bits = [r0,g0,b0,a0], high 128 bits = [r1,g1,b1,a1].
static inline __m256 LoadTwoPixels(const PixelRGBA& p0, const PixelRGBA& p1)
{
    const __m128 lo = _mm_loadu_ps(&p0.r);
    const __m128 hi = _mm_loadu_ps(&p1.r);
    return _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
}

/// Store a __m256 into a PixelRGBA_2.
static inline void StoreTwoPixels(PixelRGBA_2& dst, __m256 v)
{
    _mm_storeu_ps(&dst.p0.r, _mm256_castps256_ps128(v));
    _mm_storeu_ps(&dst.p1.r, _mm256_extractf128_ps(v, 1));
}

/// Per-lane clamp to [0, 1].
static inline __m256 Clamp01SIMD_2(__m256 v)
{
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one  = _mm256_set1_ps(1.0f);
    return _mm256_min_ps(_mm256_max_ps(v, zero), one);
}

/// Premultiply two pixels at once (same opacity for both, since it comes
/// from the same frame-source weight).
static inline __m256 PremultiplySIMD_2(__m256 pixel, float opacity)
{
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one  = _mm256_set1_ps(1.0f);

    // Spread alpha across each 128-bit lane: [a0,a0,a0,a0, a1,a1,a1,a1]
    const __m256 alpha  = _mm256_shuffle_ps(pixel, pixel, _MM_SHUFFLE(3, 3, 3, 3));
    const __m256 opV    = _mm256_set1_ps(opacity);
    const __m256 preA   = _mm256_mul_ps(alpha, opV);
    const __m256 clampA = _mm256_min_ps(_mm256_max_ps(preA, zero), one);

    // Replace alpha with 1.0 so RGB multiplies by clampedAlpha and A becomes clampedAlpha itself.
    // Blend mask 0b10001000 keeps bits 3 and 7 (alpha lanes) from `one`.
    const __m256 rgbOne = _mm256_blend_ps(pixel, one, 0b10001000);
    return _mm256_mul_ps(rgbOne, clampA);
}

/// Unpremultiply two pixels at once (returns zero for lanes where alpha <= 0).
static inline __m256 UnpremultiplySIMD_2(__m256 pixel)
{
    // Spread alpha across each 128-bit lane
    const __m256 alpha = _mm256_shuffle_ps(pixel, pixel, _MM_SHUFFLE(3, 3, 3, 3));

    const __m256 zero = _mm256_setzero_ps();
    const __m256 one  = _mm256_set1_ps(1.0f);

    // Per-lane mask: 0xFFFFFFFF where alpha <= 0, else 0
    const __m256 mask = _mm256_cmp_ps(alpha, zero, _CMP_LE_OS);

    // Avoid division by zero: use 1.0 where alpha <= 0
    const __m256 safeAlpha = _mm256_blendv_ps(alpha, one, mask);
    const __m256 invAlpha  = _mm256_div_ps(one, safeAlpha);

    // Zero the reciprocal for lanes that were <= 0
    const __m256 cleanInv = _mm256_andnot_ps(mask, invAlpha);

    // Scale RGB by invAlpha; preserve original alpha from pixel
    __m256 result = _mm256_mul_ps(pixel, cleanInv);
    result = _mm256_blend_ps(result, pixel, 0b10001000);

    return Clamp01SIMD_2(result);
}

/// Over composite for two pixels: out = src + dst * (1 - src.a)
static inline __m256 OverSIMD_2(__m256 dst, __m256 src)
{
    // Spread src.a across each 128-bit lane
    const __m256 srcA = _mm256_shuffle_ps(src, src, _MM_SHUFFLE(3, 3, 3, 3));
    const __m256 one  = _mm256_set1_ps(1.0f);
    const __m256 dstFactor = _mm256_sub_ps(one, srcA);
    return _mm256_add_ps(_mm256_mul_ps(dst, dstFactor), src);
}


// -----------------------------------------------------------------------
// ComposeTwoSamplesAVX2 - processes two pixels simultaneously with __m256
// -----------------------------------------------------------------------
PixelRGBA_2 ComposeTwoSamplesAVX2(
    const std::vector<TemporalSample>& samples0,
    const std::vector<TemporalSample>& samples1,
    BlendMode blendMode)
{
    // Both vectors must have been built from the same source list
    // so they always have equal size in practice.
    const std::size_t count = (std::min)(samples0.size(), samples1.size());
    if (count == 0)
    {
        return PixelRGBA_2{};
    }

    if (blendMode == BlendMode::Add)
    {
        __m256 sum = _mm256_setzero_ps();

        for (std::size_t i = 0; i < count; ++i)
        {
            const __m256 px = LoadTwoPixels(samples0[i].pixel, samples1[i].pixel);
            sum = _mm256_add_ps(sum, PremultiplySIMD_2(px, samples0[i].opacity));
        }

        sum = Clamp01SIMD_2(sum);
        PixelRGBA_2 result;
        StoreTwoPixels(result, UnpremultiplySIMD_2(sum));
        return result;
    }

    if (blendMode == BlendMode::Maximum || blendMode == BlendMode::Minimum)
    {
        __m256 value = PremultiplySIMD_2(
            LoadTwoPixels(samples0[0].pixel, samples1[0].pixel),
            samples0[0].opacity);

        for (std::size_t i = 1; i < count; ++i)
        {
            const __m256 premul = PremultiplySIMD_2(
                LoadTwoPixels(samples0[i].pixel, samples1[i].pixel),
                samples0[i].opacity);

            if (blendMode == BlendMode::Maximum)
            {
                value = _mm256_max_ps(value, premul);
            }
            else
            {
                value = _mm256_min_ps(value, premul);
            }
        }

        PixelRGBA_2 result;
        StoreTwoPixels(result, UnpremultiplySIMD_2(value));
        return result;
    }

    // Over blending (BlendNewOnTop / BlendNewOnBottom)
    __m256 composed = _mm256_setzero_ps();

    for (std::size_t i = 0; i < count; ++i)
    {
        const __m256 premul = PremultiplySIMD_2(
            LoadTwoPixels(samples0[i].pixel, samples1[i].pixel),
            samples0[i].opacity);

        if (blendMode == BlendMode::BlendNewOnBottom)
        {
            composed = OverSIMD_2(composed, premul);
        }
        else
        {
            composed = OverSIMD_2(premul, composed);
        }
    }

    PixelRGBA_2 result;
    StoreTwoPixels(result, UnpremultiplySIMD_2(composed));
    return result;
}

} // namespace frame_echo
