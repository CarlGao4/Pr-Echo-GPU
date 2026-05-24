#include "FrameEchoCore.h"

#include <immintrin.h>
#include <algorithm>

// -----------------------------------------------------------------------
// This file is compiled separately with /arch:AVX512 so all _mm512_*
// intrinsics are available.
//
// All helpers operate on __m512 (512-bit / 16 x float) to process FOUR
// PixelRGBA values simultaneously.
// -----------------------------------------------------------------------

namespace frame_echo
{

// =======================================================================
// AVX-512 (512-bit) helpers - process FOUR pixels (16 x float) at once.
// =======================================================================

/// Load four PixelRGBA values into a __m512 register.
/// Layout: lanes 0-3 = p0.rgba, lanes 4-7 = p1.rgba,
///         lanes 8-11 = p2.rgba, lanes 12-15 = p3.rgba.
///
/// Each pixel is loaded independently via _mm_loadu_ps, then combined
/// into one __m512 with _mm512_insertf32x4.  This is correct even when
/// the four pixels are NOT contiguous in memory (e.g. coming from four
/// separate TemporalSample vectors).
static inline __m512 LoadFourPixels(
    const PixelRGBA& p0, const PixelRGBA& p1,
    const PixelRGBA& p2, const PixelRGBA& p3)
{
    const __m128 v0 = _mm_loadu_ps(&p0.r);
    const __m128 v1 = _mm_loadu_ps(&p1.r);
    const __m128 v2 = _mm_loadu_ps(&p2.r);
    const __m128 v3 = _mm_loadu_ps(&p3.r);
    __m512 r = _mm512_castps128_ps512(v0);
    r = _mm512_insertf32x4(r, v1, 1);
    r = _mm512_insertf32x4(r, v2, 2);
    r = _mm512_insertf32x4(r, v3, 3);
    return r;
}

/// Store a __m512 into a PixelRGBA_4.  Each 128-bit lane is extracted
/// and stored independently (the dst is always a contiguous PixelRGBA_4,
/// but this avoids relying on that assumption implicitly).
static inline void StoreFourPixels(PixelRGBA_4& dst, __m512 v)
{
    _mm_storeu_ps(&dst.p0.r, _mm512_castps512_ps128(v));
    _mm_storeu_ps(&dst.p1.r, _mm512_extractf32x4_ps(v, 1));
    _mm_storeu_ps(&dst.p2.r, _mm512_extractf32x4_ps(v, 2));
    _mm_storeu_ps(&dst.p3.r, _mm512_extractf32x4_ps(v, 3));
}

/// Per-lane clamp to [0, 1].
static inline __m512 Clamp01SIMD_4(__m512 v)
{
    const __m512 zero = _mm512_setzero_ps();
    const __m512 one  = _mm512_set1_ps(1.0f);
    return _mm512_min_ps(_mm512_max_ps(v, zero), one);
}

/// Premultiply four pixels at once (same opacity for all).
static inline __m512 PremultiplySIMD_4(__m512 pixel, float opacity)
{
    const __m512 zero = _mm512_setzero_ps();
    const __m512 one  = _mm512_set1_ps(1.0f);
    const __m512 opV  = _mm512_set1_ps(opacity);

    // Shuffle alpha within each 128-bit lane: lane0 [a0,a0,a0,a0], lane1 [a1,a1,a1,a1], etc.
    // _MM_SHUFFLE(3,3,3,3) works within each 128-bit lane.
    const __m512 alpha  = _mm512_shuffle_ps(pixel, pixel, _MM_SHUFFLE(3, 3, 3, 3));
    const __m512 preA   = _mm512_mul_ps(alpha, opV);
    const __m512 clampA = _mm512_min_ps(_mm512_max_ps(preA, zero), one);

    // Blend mask: set bit 3, 7, 11, 15 to take alpha from `one` (value 1.0f).
    const __m512 rgbOne = _mm512_mask_blend_ps(0b1000100010001000, pixel, one);
    return _mm512_mul_ps(rgbOne, clampA);
}

/// Unpremultiply four pixels at once (returns zero for lanes where alpha <= 0).
static inline __m512 UnpremultiplySIMD_4(__m512 pixel)
{
    const __m512 zero = _mm512_setzero_ps();
    const __m512 one  = _mm512_set1_ps(1.0f);

    // Spread alpha across each 128-bit lane
    const __m512 alpha = _mm512_shuffle_ps(pixel, pixel, _MM_SHUFFLE(3, 3, 3, 3));

    // Compare mask: 0xFFFF per lane where alpha <= 0
    const __mmask16 mask = _mm512_cmp_ps_mask(alpha, zero, _CMP_LE_OS);

    // Safe reciprocal: use 1.0 where alpha <= 0
    const __m512 safeAlpha = _mm512_mask_blend_ps(mask, alpha, one);
    const __m512 invAlpha  = _mm512_div_ps(one, safeAlpha);

    // Zero the reciprocal for lanes that were <= 0
    const __m512 cleanInv = _mm512_maskz_mov_ps(~mask, invAlpha);

    // Scale RGB by invAlpha; preserve original alpha from pixel
    __m512 result = _mm512_mul_ps(pixel, cleanInv);
    result = _mm512_mask_blend_ps(0b1000100010001000, result, pixel);

    return Clamp01SIMD_4(result);
}

/// Over composite for four pixels: out = src + dst * (1 - src.a)
static inline __m512 OverSIMD_4(__m512 dst, __m512 src)
{
    const __m512 one  = _mm512_set1_ps(1.0f);
    const __m512 srcA = _mm512_shuffle_ps(src, src, _MM_SHUFFLE(3, 3, 3, 3));
    const __m512 dstFactor = _mm512_sub_ps(one, srcA);
    return _mm512_fmadd_ps(dst, dstFactor, src);
}


// -----------------------------------------------------------------------
// ComposeFourSamplesAVX512 - processes four pixels simultaneously with __m512
// -----------------------------------------------------------------------
PixelRGBA_4 ComposeFourSamplesAVX512(
    const std::vector<TemporalSample>& samples0,
    const std::vector<TemporalSample>& samples1,
    const std::vector<TemporalSample>& samples2,
    const std::vector<TemporalSample>& samples3,
    BlendMode blendMode)
{
    const std::size_t count = (std::min)({samples0.size(), samples1.size(),
                                          samples2.size(), samples3.size()});
    if (count == 0)
    {
        return PixelRGBA_4{};
    }

    if (blendMode == BlendMode::Add)
    {
        __m512 sum = _mm512_setzero_ps();

        for (std::size_t i = 0; i < count; ++i)
        {
            const __m512 px = LoadFourPixels(
                samples0[i].pixel, samples1[i].pixel,
                samples2[i].pixel, samples3[i].pixel);
            sum = _mm512_add_ps(sum, PremultiplySIMD_4(px, samples0[i].opacity));
        }

        sum = Clamp01SIMD_4(sum);
        PixelRGBA_4 result;
        StoreFourPixels(result, UnpremultiplySIMD_4(sum));
        return result;
    }

    if (blendMode == BlendMode::Maximum || blendMode == BlendMode::Minimum)
    {
        __m512 value = PremultiplySIMD_4(
            LoadFourPixels(samples0[0].pixel, samples1[0].pixel,
                           samples2[0].pixel, samples3[0].pixel),
            samples0[0].opacity);

        for (std::size_t i = 1; i < count; ++i)
        {
            const __m512 premul = PremultiplySIMD_4(
                LoadFourPixels(samples0[i].pixel, samples1[i].pixel,
                               samples2[i].pixel, samples3[i].pixel),
                samples0[i].opacity);

            if (blendMode == BlendMode::Maximum)
            {
                value = _mm512_max_ps(value, premul);
            }
            else
            {
                value = _mm512_min_ps(value, premul);
            }
        }

        PixelRGBA_4 result;
        StoreFourPixels(result, UnpremultiplySIMD_4(value));
        return result;
    }

    // Over blending (BlendNewOnTop / BlendNewOnBottom)
    __m512 composed = _mm512_setzero_ps();

    for (std::size_t i = 0; i < count; ++i)
    {
        const __m512 premul = PremultiplySIMD_4(
            LoadFourPixels(samples0[i].pixel, samples1[i].pixel,
                           samples2[i].pixel, samples3[i].pixel),
            samples0[i].opacity);

        if (blendMode == BlendMode::BlendNewOnBottom)
        {
            composed = OverSIMD_4(composed, premul);
        }
        else
        {
            composed = OverSIMD_4(premul, composed);
        }
    }

    PixelRGBA_4 result;
    StoreFourPixels(result, UnpremultiplySIMD_4(composed));
    return result;
}

} // namespace frame_echo
