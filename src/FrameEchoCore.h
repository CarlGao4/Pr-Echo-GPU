#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace frame_echo
{
// -----------------------------------------------------------------------
// CPU feature detection - set by DetectCPUFeatures(), read by ComposeSamples.
// Detects SSE4.2, AVX2, and AVX-512 support via CPUID.
// -----------------------------------------------------------------------
extern bool g_hasSSE4_2;
extern bool g_hasAVX2;
extern bool g_hasAVX512;
void DetectCPUFeatures();

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

/// Holds two PixelRGBA values in AoS layout for AVX2 (256-bit / 8 x float) processing.
/// Memory layout: [r0, g0, b0, a0, r1, g1, b1, a1] - identical to storing two
/// PixelRGBA objects adjacently, but with 32-byte alignment for AVX2 loads/stores.
struct alignas(32) PixelRGBA_2
{
    PixelRGBA p0;
    PixelRGBA p1;
};

/// Holds four PixelRGBA values for AVX-512 (512-bit / 16 x float) processing.
/// Memory layout: [p0.rgba, p1.rgba, p2.rgba, p3.rgba] contiguous.
struct alignas(64) PixelRGBA_4
{
    PixelRGBA p0;
    PixelRGBA p1;
    PixelRGBA p2;
    PixelRGBA p3;
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

// SSE2-accelerated dispatch target (compiled with /arch:AVX2 for VEX encoding)
PixelRGBA ComposeSamples_SSE2(const std::vector<TemporalSample>& samples, BlendMode blendMode);

// AVX2-accelerated dispatch target (processes 2 pixels at once using _mm256_*)
PixelRGBA ComposeSamples_AVX2(const std::vector<TemporalSample>& samples, BlendMode blendMode);

// AVX2 batch: composes two pixels' sample sets simultaneously using 256-bit intrinsics.
// samples0 and samples1 must have the same size; corresponding entries share the same
// opacity/weight (only pixel values differ between the two vectors).
PixelRGBA_2 ComposeTwoSamplesAVX2(
    const std::vector<TemporalSample>& samples0,
    const std::vector<TemporalSample>& samples1,
    BlendMode blendMode);

// AVX-512 batch: composes four pixels' sample sets simultaneously using 512-bit intrinsics.
PixelRGBA_4 ComposeFourSamplesAVX512(
    const std::vector<TemporalSample>& samples0,
    const std::vector<TemporalSample>& samples1,
    const std::vector<TemporalSample>& samples2,
    const std::vector<TemporalSample>& samples3,
    BlendMode blendMode);

} // namespace frame_echo
