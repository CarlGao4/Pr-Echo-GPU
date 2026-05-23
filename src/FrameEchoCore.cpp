#include "FrameEchoCore.h"

namespace frame_echo
{

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

static std::vector<float> BuildDirectionalWeights(
    int frameCount,
    WindowFunction function,
    float falloffRatio,
    WindowFalloffMapping mapping,
    float maxOpacity,
    bool reverseOrder)
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
        const int orderedIndex = reverseOrder ? (frameCount - 1 - index) : index;
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

std::vector<float> BuildBackwardWeights(
    int frameCount,
    WindowFunction function,
    float falloffRatio,
    WindowFalloffMapping mapping,
    float maxOpacity)
{
    return BuildDirectionalWeights(frameCount, function, falloffRatio, mapping, maxOpacity, false);
}

std::vector<float> BuildForwardWeights(
    int frameCount,
    WindowFunction function,
    float falloffRatio,
    WindowFalloffMapping mapping,
    float maxOpacity)
{
    return BuildDirectionalWeights(frameCount, function, falloffRatio, mapping, maxOpacity, true);
}

float GetCurrentFrameOpacity(const TemporalSettings& settings)
{
    const float forward = Clamp01(settings.forwardMaxOpacity);
    const float backward = Clamp01(settings.backwardMaxOpacity);

    return 0.5f * (forward + backward);
}

PixelRGBA Premultiply(const PixelRGBA& pixel, float opacity)
{
    const float alpha = Clamp01(pixel.a * opacity);
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

    const float inverseAlpha = 1.0f / pixel.a;
    return PixelRGBA{
        Clamp01(pixel.r * inverseAlpha),
        Clamp01(pixel.g * inverseAlpha),
        Clamp01(pixel.b * inverseAlpha),
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
    if (samples.empty())
    {
        return PixelRGBA{};
    }

    if (blendMode == BlendMode::Add)
    {
        PixelRGBA sum{};
        for (const TemporalSample& sample : samples)
        {
            const PixelRGBA premultiplied = Premultiply(sample.pixel, sample.opacity);
            sum.r += premultiplied.r;
            sum.g += premultiplied.g;
            sum.b += premultiplied.b;
            sum.a += premultiplied.a;
        }

        sum.r = Clamp01(sum.r);
        sum.g = Clamp01(sum.g);
        sum.b = Clamp01(sum.b);
        sum.a = Clamp01(sum.a);
        return Unpremultiply(sum);
    }

    if (blendMode == BlendMode::Maximum || blendMode == BlendMode::Minimum)
    {
        PixelRGBA value = Premultiply(samples.front().pixel, samples.front().opacity);
        for (std::size_t index = 1; index < samples.size(); ++index)
        {
            const PixelRGBA premultiplied = Premultiply(samples[index].pixel, samples[index].opacity);
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
            Over(Premultiply(sample.pixel, sample.opacity), composed):
            Over(composed, Premultiply(sample.pixel, sample.opacity));
    }

    return Unpremultiply(composed);
}

} // namespace frame_echo
