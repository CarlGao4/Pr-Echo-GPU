#include "FrameEchoCore.h"

#include <intrin.h>

namespace frame_echo
{

// -----------------------------------------------------------------------
// CPU feature detection via CPUID - detects SSE4.2, AVX2, and AVX-512.
// -----------------------------------------------------------------------
bool g_hasSSE4_2 = false;
bool g_hasAVX2 = false;
bool g_hasAVX512 = false;

void DetectCPUFeatures()
{
    int cpuInfo[4] = {};

    // Default all flags to false
    g_hasSSE4_2 = false;
    g_hasAVX2 = false;
    g_hasAVX512 = false;

    // CPUID leaf 0: get max supported leaf
    __cpuid(cpuInfo, 0);
    const int maxLeaf = cpuInfo[0];
    if (maxLeaf < 1)
    {
        return;  // Leaf 1 not available - nothing to detect
    }

    // Leaf 1: check SSE4.2 (ECX bit 20) and XSAVE (ECX bit 27)
    __cpuid(cpuInfo, 1);
    g_hasSSE4_2 = (cpuInfo[2] & (1 << 20)) != 0;
    const bool hasXSAVE = (cpuInfo[2] & (1 << 27)) != 0;

    if (maxLeaf < 7)
    {
        return;  // Leaf 7 not available - AVX2/AVX-512 can't be checked
    }

    // Leaf 7 subleaf 0: check AVX2 and AVX-512 feature bits
    __cpuidex(cpuInfo, 7, 0);
    // EBX bit 5 = AVX2
    g_hasAVX2 = (cpuInfo[1] & (1 << 5)) != 0;

    // EBX bit 16 = AVX512F
    const bool hasAVX512F = (cpuInfo[1] & (1 << 16)) != 0;
    // EBX bit 17 = AVX512DQ
    const bool hasAVX512DQ = (cpuInfo[1] & (1 << 17)) != 0;

    if (hasXSAVE && hasAVX512F && hasAVX512DQ)
    {
        // Verify OS has enabled the upper ZMM registers (XCR0 bits 5, 6, 7)
        unsigned long long xcr0 = _xgetbv(0);
        g_hasAVX512 = ((xcr0 & 0xE0) == 0xE0);  // bits 5, 6, 7 all set
    }
}

float Clamp01(float value)
{
    return std::max(0.0f, std::min(1.0f, value));
}

PixelRGBA MakeBlankTransparent()
{
    return PixelRGBA{};
}

float EvaluateWindowFunction(WindowFunction function, float normalizedPosition)
{
    const float t = Clamp01(normalizedPosition);

    switch (function)
    {
    case WindowFunction::Constant:
        return 1.0f;
    case WindowFunction::Linear:
        return t;
    case WindowFunction::SquareRoot:
        return std::sqrt(t);
    case WindowFunction::Square:
        return t * t;
    case WindowFunction::Cosine:
        return 0.5f - 0.5f * std::cos(3.14159265358979323846f * t);
    case WindowFunction::SmoothStep:
        return t * t * (3.0f - 2.0f * t);
    }

    return t;
}

// ---- Popup-to-enum conversion helpers ----

WindowFunction ToWindowFunction(int popupValue)
{
    switch (popupValue)
    {
    case FRAME_ECHO_WINDOW_CONSTANT:
        return WindowFunction::Constant;
    case FRAME_ECHO_WINDOW_LINEAR:
        return WindowFunction::Linear;
    case FRAME_ECHO_WINDOW_SQUARE_ROOT:
        return WindowFunction::SquareRoot;
    case FRAME_ECHO_WINDOW_SQUARE:
        return WindowFunction::Square;
    case FRAME_ECHO_WINDOW_COSINE:
        return WindowFunction::Cosine;
    case FRAME_ECHO_WINDOW_SMOOTH_STEP:
    default:
        return WindowFunction::SmoothStep;
    }
}

BlendMode ToBlendMode(int popupValue)
{
    switch (popupValue)
    {
    case FRAME_ECHO_BLEND_NEW_ON_BOTTOM:
        return BlendMode::BlendNewOnBottom;
    case FRAME_ECHO_BLEND_ADD:
        return BlendMode::Add;
    case FRAME_ECHO_BLEND_MAXIMUM:
        return BlendMode::Maximum;
    case FRAME_ECHO_BLEND_MINIMUM:
        return BlendMode::Minimum;
    case FRAME_ECHO_BLEND_NEW_ON_TOP:
    default:
        return BlendMode::BlendNewOnTop;
    }
}

FrameShortageBehavior ToFrameShortageBehavior(int popupValue)
{
    switch (popupValue)
    {
    case FRAME_ECHO_FRAME_SHORTAGE_REPEAT:
        return FrameShortageBehavior::Repeat;
    case FRAME_ECHO_FRAME_SHORTAGE_SOLID_COLOR:
        return FrameShortageBehavior::SolidColor;
    case FRAME_ECHO_FRAME_SHORTAGE_BLANK_TRANSPARENT:
    default:
        return FrameShortageBehavior::BlankTransparent;
    }
}

std::vector<float> BuildDirectionalWeights(
    int frameCount,
    WindowFunction function,
    float falloffRatio,
    WindowFalloffMapping mapping,
    float maxOpacity)
{
    std::vector<float> weights;
    if (frameCount <= 0)
    {
        return weights;
    }

    weights.reserve(static_cast<std::size_t>(frameCount));
    const float clampedMaxOpacity = Clamp01(maxOpacity);
    const float clampedMinimum = Clamp01(falloffRatio);

    for (int index = 0; index < frameCount; ++index)
    {
        const int orderedIndex = frameCount - 1 - index;
        const float baseNormalized = static_cast<float>(orderedIndex + 1) / static_cast<float>(frameCount + 1);
        float mappedInput = baseNormalized;
        float weight = 0.0f;

        if (mapping == WindowFalloffMapping::InputMapped)
        {
            mappedInput = Clamp01(baseNormalized * (1.0f - clampedMinimum));
            weight = EvaluateWindowFunction(function, mappedInput);
        }
        else
        {
            const float evaluated = EvaluateWindowFunction(function, mappedInput);
            weight = (evaluated * (1.0f - clampedMinimum)) + clampedMinimum;
        }

        weights.push_back(weight * clampedMaxOpacity);
    }

    return weights;
}

float GetCurrentFrameWeight(const TemporalSettings& settings)
{
    const float forward = Clamp01(settings.forwardMaxOpacity);
    const float backward = Clamp01(settings.backwardMaxOpacity);

    return 0.5f * (forward + backward);
}

PixelRGBA Premultiply(const PixelRGBA& pixel, float weight)
{
    const float alpha = Clamp01(pixel.a * weight);
    return PixelRGBA{
        pixel.r * alpha,
        pixel.g * alpha,
        pixel.b * alpha,
        alpha,
    };
}

PixelRGBA Unpremultiply(const PixelRGBA& pixel)
{
    if (pixel.a <= 0.0f)
    {
        return PixelRGBA{};
    }

    const float inverseAlpha = 1.0f / Clamp01(pixel.a);
    return PixelRGBA{
        pixel.r * inverseAlpha,
        pixel.g * inverseAlpha,
        pixel.b * inverseAlpha,
        Clamp01(pixel.a),
    };
}

static PixelRGBA Over(const PixelRGBA& dst, const PixelRGBA& src)
{
    const float dstFactor = 1.0f - src.a;
    return PixelRGBA{
        src.r + dst.r * dstFactor,
        src.g + dst.g * dstFactor,
        src.b + dst.b * dstFactor,
        src.a + dst.a * dstFactor,
    };
}

PixelRGBA ComposeSamples(const std::vector<TemporalSample>& samples, BlendMode blendMode)
{
    // Dispatch priority: AVX-512 > AVX2 > SSE4.2 > scalar fallback
    // AVX-512 and AVX2 versions should be called with multiple samples, so only call SSE2 version as only 1 sample is provided here
    // For actual usage, check RenderRows function
    if (g_hasSSE4_2)
    {
        return ComposeSamples_SSE2(samples, blendMode);
    }

    if (samples.empty())
    {
        return PixelRGBA{};
    }

    if (blendMode == BlendMode::Add)
    {
        PixelRGBA sum{};
        for (const TemporalSample& sample : samples)
        {
            const PixelRGBA premultiplied = Premultiply(sample.pixel, sample.weight);
            sum.r += premultiplied.r;
            sum.g += premultiplied.g;
            sum.b += premultiplied.b;
            sum.a += premultiplied.a;
        }

        return Unpremultiply(sum);
    }

    if (blendMode == BlendMode::Maximum || blendMode == BlendMode::Minimum)
    {
        PixelRGBA value = Premultiply(samples.front().pixel, samples.front().weight);
        for (std::size_t index = 1; index < samples.size(); ++index)
        {
            const PixelRGBA premultiplied = Premultiply(samples[index].pixel, samples[index].weight);
            if (blendMode == BlendMode::Maximum)
            {
                value.r = std::max(value.r, premultiplied.r);
                value.g = std::max(value.g, premultiplied.g);
                value.b = std::max(value.b, premultiplied.b);
                value.a = std::max(value.a, premultiplied.a);
            }
            else
            {
                value.r = std::min(value.r, premultiplied.r);
                value.g = std::min(value.g, premultiplied.g);
                value.b = std::min(value.b, premultiplied.b);
                value.a = std::min(value.a, premultiplied.a);
            }
        }

        return Unpremultiply(value);
    }

    PixelRGBA composed{};
    for (const TemporalSample& sample : samples)
    {
        composed = blendMode == BlendMode::BlendNewOnBottom ?
            Over(Premultiply(sample.pixel, sample.weight), composed):
            Over(composed, Premultiply(sample.pixel, sample.weight));
    }

    return Unpremultiply(composed);
}

} // namespace frame_echo
