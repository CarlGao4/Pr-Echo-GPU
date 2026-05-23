#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace frame_echo
{
enum class BlendMode : std::uint32_t
{
    BlendNewOnTop = 0,
    BlendNewOnBottom = 1,
    Add = 2,
    Maximum = 3,
    Minimum = 4,
};

enum class WindowFunction : std::uint32_t
{
    Constant = 0,
    Linear = 1,
    SquareRoot = 2,
    Square = 3,
    Cosine = 4,
    SmoothStep = 5,
};

enum class WindowFalloffMapping : std::uint32_t
{
    OutputMapped = 0,
    InputMapped = 1,
};

enum class FrameShortageBehavior : std::uint32_t
{
    BlankTransparent = 0,
    Repeat = 1,
    SolidColor = 2,
};

struct PixelRGBA
{
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 0.0f;
};

struct TemporalSample
{
    PixelRGBA pixel;
    float opacity = 1.0f;
};

struct TemporalSettings
{
    int backwardFrameCount = 0;
    int forwardFrameCount = 0;
    BlendMode blendMode = BlendMode::BlendNewOnTop;
    WindowFunction forwardWindowFunction = WindowFunction::Linear;
    WindowFunction backwardWindowFunction = WindowFunction::Linear;
    float forwardWindowFalloffRatio = 1.0f;
    float backwardWindowFalloffRatio = 1.0f;
    WindowFalloffMapping forwardWindowFalloffMapping = WindowFalloffMapping::OutputMapped;
    WindowFalloffMapping backwardWindowFalloffMapping = WindowFalloffMapping::OutputMapped;
    float forwardMaxOpacity = 1.0f;
    float backwardMaxOpacity = 1.0f;
    bool syncForwardBackward = true;
    FrameShortageBehavior frameShortageBehavior = FrameShortageBehavior::BlankTransparent;
};

float Clamp01(float value);
PixelRGBA MakeBlankTransparent();
float EvaluateWindowFunction(WindowFunction function, float normalizedPosition);
std::vector<float> BuildBackwardWeights(
    int frameCount,
    WindowFunction function,
    float falloffRatio,
    WindowFalloffMapping mapping,
    float maxOpacity);
std::vector<float> BuildForwardWeights(
    int frameCount,
    WindowFunction function,
    float falloffRatio,
    WindowFalloffMapping mapping,
    float maxOpacity);
float GetCurrentFrameOpacity(const TemporalSettings& settings);
PixelRGBA Premultiply(const PixelRGBA& pixel, float opacity);
PixelRGBA Unpremultiply(const PixelRGBA& pixel);
PixelRGBA ComposeSamples(const std::vector<TemporalSample>& samples, BlendMode blendMode);

} // namespace frame_echo
