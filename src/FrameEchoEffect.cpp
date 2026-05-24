#include "FrameEchoCore.h"
#include "FrameEchoParams.h"

#include "AEConfig.h"
#include "PrSDKTypes.h"
#include "AE_Effect.h"
#include "AE_EffectSuites.h"
#include "A.h"
#include "AE_Macros.h"
#include "AEFX_SuiteHandlerTemplate.h"
#include "Param_Utils.h"

#include "PrSDKAESupport.h"

#include <thread>
#include <vector>

static const char kBlendModePopup[] =
    "Blend (New On Top)|"
    "Blend (New On Bottom)|"
    "Add|"
    "Maximum|"
    "Minimum";

static const char kWindowFunctionPopup[] =
    "Constant|"
    "Linear|"
    "Square Root|"
    "Square|"
    "Cosine|"
    "Smooth Step";

static const char kWindowFalloffMappingPopup[] =
    "Output mapped|"
    "Input mapped";

static const char kFrameShortagePopup[] =
    "Blank (Transparent)|"
    "Repeat|"
    "Solid Color";

static constexpr A_long kCountSliderMax = 300;

bool isPremiere = false;

static frame_echo::WindowFunction ToWindowFunction(A_long popupValue)
{
    switch (popupValue)
    {
    case FRAME_ECHO_WINDOW_CONSTANT:
        return frame_echo::WindowFunction::Constant;
    case FRAME_ECHO_WINDOW_LINEAR:
        return frame_echo::WindowFunction::Linear;
    case FRAME_ECHO_WINDOW_SQUARE_ROOT:
        return frame_echo::WindowFunction::SquareRoot;
    case FRAME_ECHO_WINDOW_SQUARE:
        return frame_echo::WindowFunction::Square;
    case FRAME_ECHO_WINDOW_COSINE:
        return frame_echo::WindowFunction::Cosine;
    case FRAME_ECHO_WINDOW_SMOOTH_STEP:
    default:
        return frame_echo::WindowFunction::SmoothStep;
    }
}

static frame_echo::BlendMode ToBlendMode(A_long popupValue)
{
    switch (popupValue)
    {
    case FRAME_ECHO_BLEND_NEW_ON_BOTTOM:
        return frame_echo::BlendMode::BlendNewOnBottom;
    case FRAME_ECHO_BLEND_ADD:
        return frame_echo::BlendMode::Add;
    case FRAME_ECHO_BLEND_MAXIMUM:
        return frame_echo::BlendMode::Maximum;
    case FRAME_ECHO_BLEND_MINIMUM:
        return frame_echo::BlendMode::Minimum;
    case FRAME_ECHO_BLEND_NEW_ON_TOP:
    default:
        return frame_echo::BlendMode::BlendNewOnTop;
    }
}

static frame_echo::FrameShortageBehavior ToFrameShortageBehavior(A_long popupValue)
{
    switch (popupValue)
    {
    case FRAME_ECHO_FRAME_SHORTAGE_REPEAT:
        return frame_echo::FrameShortageBehavior::Repeat;
    case FRAME_ECHO_FRAME_SHORTAGE_SOLID_COLOR:
        return frame_echo::FrameShortageBehavior::SolidColor;
    case FRAME_ECHO_FRAME_SHORTAGE_BLANK_TRANSPARENT:
    default:
        return frame_echo::FrameShortageBehavior::BlankTransparent;
    }
}

static frame_echo::TemporalSettings BuildTemporalSettings(PF_ParamDef* params[])
{
    frame_echo::TemporalSettings settings;
    settings.backwardFrameCount = params[FRAME_ECHO_BACKWARD_FRAME_COUNT]->u.sd.value;
    settings.forwardFrameCount = params[FRAME_ECHO_FORWARD_FRAME_COUNT]->u.sd.value;
    settings.blendMode = ToBlendMode(params[FRAME_ECHO_BLEND_MODE]->u.pd.value);
    settings.syncForwardBackward = params[FRAME_ECHO_SYNC_FORWARD_BACKWARD]->u.bd.value != FALSE;
    settings.forwardWindowFunction = ToWindowFunction(params[FRAME_ECHO_FORWARD_WINDOW_FUNCTION]->u.pd.value);
    settings.backwardWindowFunction = ToWindowFunction(params[FRAME_ECHO_BACKWARD_WINDOW_FUNCTION]->u.pd.value);
    settings.forwardWindowFalloffRatio = params[FRAME_ECHO_FORWARD_WINDOW_FALLOFF_RATIO]->u.fs_d.value;
    settings.backwardWindowFalloffRatio = params[FRAME_ECHO_BACKWARD_WINDOW_FALLOFF_RATIO]->u.fs_d.value;
    settings.forwardWindowFalloffMapping =
        params[FRAME_ECHO_FORWARD_WINDOW_FALLOFF_MAPPING]->u.pd.value == FRAME_ECHO_WINDOW_FALLOFF_INPUT_MAPPED
        ? frame_echo::WindowFalloffMapping::InputMapped
        : frame_echo::WindowFalloffMapping::OutputMapped;
    settings.backwardWindowFalloffMapping =
        params[FRAME_ECHO_BACKWARD_WINDOW_FALLOFF_MAPPING]->u.pd.value == FRAME_ECHO_WINDOW_FALLOFF_INPUT_MAPPED
        ? frame_echo::WindowFalloffMapping::InputMapped
        : frame_echo::WindowFalloffMapping::OutputMapped;
    settings.forwardMaxOpacity = params[FRAME_ECHO_FORWARD_MAX_OPACITY]->u.fs_d.value;
    settings.backwardMaxOpacity = params[FRAME_ECHO_BACKWARD_MAX_OPACITY]->u.fs_d.value;
    settings.frameShortageBehavior = ToFrameShortageBehavior(params[FRAME_ECHO_FRAME_SHORTAGE_BEHAVIOR]->u.pd.value);
    return settings;
}

static void UpdateSyncUiState(PF_ParamDef* params[])
{
    const bool syncEnabled = params[FRAME_ECHO_SYNC_FORWARD_BACKWARD]->u.bd.value != FALSE;

    if (syncEnabled)
    {
        params[FRAME_ECHO_BACKWARD_WINDOW_FUNCTION]->ui_flags |= PF_PUI_DISABLED;
        params[FRAME_ECHO_BACKWARD_WINDOW_FALLOFF_RATIO]->ui_flags |= PF_PUI_DISABLED;
        params[FRAME_ECHO_BACKWARD_WINDOW_FALLOFF_MAPPING]->ui_flags |= PF_PUI_DISABLED;
        params[FRAME_ECHO_BACKWARD_MAX_OPACITY]->ui_flags |= PF_PUI_DISABLED;
    }
    else
    {
        params[FRAME_ECHO_BACKWARD_WINDOW_FUNCTION]->ui_flags &= ~PF_PUI_DISABLED;
        params[FRAME_ECHO_BACKWARD_WINDOW_FALLOFF_RATIO]->ui_flags &= ~PF_PUI_DISABLED;
        params[FRAME_ECHO_BACKWARD_WINDOW_FALLOFF_MAPPING]->ui_flags &= ~PF_PUI_DISABLED;
        params[FRAME_ECHO_BACKWARD_MAX_OPACITY]->ui_flags &= ~PF_PUI_DISABLED;
    }
}

static void UpdateShortageColorUiState(PF_ParamDef* params[])
{
    const A_long shortageMode = params[FRAME_ECHO_FRAME_SHORTAGE_BEHAVIOR]->u.pd.value;
    const bool enableColor = shortageMode == FRAME_ECHO_FRAME_SHORTAGE_SOLID_COLOR;
    if (enableColor)
    {
        params[FRAME_ECHO_FRAME_SHORTAGE_COLOR]->ui_flags &= ~PF_PUI_DISABLED;
    }
    else
    {
        params[FRAME_ECHO_FRAME_SHORTAGE_COLOR]->ui_flags |= PF_PUI_DISABLED;
    }
}

static void ApplySyncValues(PF_ParamDef* params[])
{
    const bool syncEnabled = params[FRAME_ECHO_SYNC_FORWARD_BACKWARD]->u.bd.value != FALSE;
    if (!syncEnabled)
    {
        return;
    }

    params[FRAME_ECHO_BACKWARD_WINDOW_FUNCTION]->u.pd.value = params[FRAME_ECHO_FORWARD_WINDOW_FUNCTION]->u.pd.value;
    params[FRAME_ECHO_BACKWARD_WINDOW_FALLOFF_RATIO]->u.fs_d.value = params[FRAME_ECHO_FORWARD_WINDOW_FALLOFF_RATIO]->u.fs_d.value;
    params[FRAME_ECHO_BACKWARD_WINDOW_FALLOFF_MAPPING]->u.pd.value = params[FRAME_ECHO_FORWARD_WINDOW_FALLOFF_MAPPING]->u.pd.value;
    params[FRAME_ECHO_BACKWARD_MAX_OPACITY]->u.fs_d.value = params[FRAME_ECHO_FORWARD_MAX_OPACITY]->u.fs_d.value;

    params[FRAME_ECHO_BACKWARD_WINDOW_FUNCTION]->uu.change_flags |= PF_ChangeFlag_CHANGED_VALUE;
    params[FRAME_ECHO_BACKWARD_WINDOW_FALLOFF_RATIO]->uu.change_flags |= PF_ChangeFlag_CHANGED_VALUE;
    params[FRAME_ECHO_BACKWARD_WINDOW_FALLOFF_MAPPING]->uu.change_flags |= PF_ChangeFlag_CHANGED_VALUE;
    params[FRAME_ECHO_BACKWARD_MAX_OPACITY]->uu.change_flags |= PF_ChangeFlag_CHANGED_VALUE;
}

static PF_Err UpdateParamUI(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_ParamIndex index)
{
    AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtils(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data);
    return paramUtils->PF_UpdateParamUI(in_data->effect_ref, index, params[index]);
}

static PF_Err RefreshParamUI(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[])
{
    UpdateSyncUiState(params);
    UpdateShortageColorUiState(params);

    PF_Err err = PF_Err_NONE;
    ERR(UpdateParamUI(in_data, out_data, params, FRAME_ECHO_BACKWARD_WINDOW_FUNCTION));
    ERR(UpdateParamUI(in_data, out_data, params, FRAME_ECHO_BACKWARD_WINDOW_FALLOFF_RATIO));
    ERR(UpdateParamUI(in_data, out_data, params, FRAME_ECHO_BACKWARD_WINDOW_FALLOFF_MAPPING));
    ERR(UpdateParamUI(in_data, out_data, params, FRAME_ECHO_BACKWARD_MAX_OPACITY));
    ERR(UpdateParamUI(in_data, out_data, params, FRAME_ECHO_FRAME_SHORTAGE_COLOR));
    return err;
}

static PF_Err GlobalSetup(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    // Detect CPU features (SSE4.2, AVX2, AVX-512) at startup
    frame_echo::DetectCPUFeatures();

    out_data->my_version = PF_VERSION(1, 0, 0, PF_Stage_DEVELOP, 0);
    out_data->out_flags |= PF_OutFlag_DEEP_COLOR_AWARE;
    out_data->out_flags |= PF_OutFlag_WIDE_TIME_INPUT;
    out_data->out_flags |= PF_OutFlag_NON_PARAM_VARY;
    out_data->out_flags |= PF_OutFlag_SEND_UPDATE_PARAMS_UI;
    out_data->out_flags2 |= PF_OutFlag2_PRESERVES_FULLY_OPAQUE_PIXELS;
    out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_THREADED_RENDERING;

    if (in_data->appl_id == 'PrMr')
    {
        isPremiere = true;
        AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data);
        (*pixelFormatSuite->ClearSupportedPixelFormats)(in_data->effect_ref);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
        AEFX_SuiteScoper<PF_UtilitySuite13> utilitySuite(in_data, kPFUtilitySuite, kPFUtilitySuiteVersion13, out_data);
        (*utilitySuite->EffectWantsCheckedOutFramesToMatchRenderPixelFormat)(in_data->effect_ref);
    }
    else if (in_data->appl_id != 'FXTC')
    {
        return PF_Err_INVALID_INDEX;
    }

    return PF_Err_NONE;
}

static PF_Err GlobalSetdown(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    return PF_Err_NONE;
}

static PF_Err ParamsSetup(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_ParamDef def;

    AEFX_CLR_STRUCT(def);
    PF_ADD_SLIDER(
        "Backward frame count",
        0,
        kCountSliderMax,
        0,
        kCountSliderMax,
        0,
        FRAME_ECHO_BACKWARD_FRAME_COUNT);

    AEFX_CLR_STRUCT(def);
    PF_ADD_SLIDER(
        "Forward frame count",
        0,
        kCountSliderMax,
        0,
        kCountSliderMax,
        0,
        FRAME_ECHO_FORWARD_FRAME_COUNT);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUPX(
        "Blend mode",
        5,
        FRAME_ECHO_BLEND_NEW_ON_TOP,
        kBlendModePopup,
        0,
        FRAME_ECHO_BLEND_MODE);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX(
        "Sync forward/backward settings",
        TRUE,
        PF_ParamFlag_SUPERVISE,
        FRAME_ECHO_SYNC_FORWARD_BACKWARD);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Forward max opacity",
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        PF_Precision_THOUSANDTHS,
        0,
        PF_ParamFlag_SUPERVISE,
        FRAME_ECHO_FORWARD_MAX_OPACITY);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Backward max opacity",
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        PF_Precision_THOUSANDTHS,
        0,
        0,
        FRAME_ECHO_BACKWARD_MAX_OPACITY);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUPX(
        "Forward window function",
        6,
        FRAME_ECHO_WINDOW_LINEAR,
        kWindowFunctionPopup,
        PF_ParamFlag_SUPERVISE,
        FRAME_ECHO_FORWARD_WINDOW_FUNCTION);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUPX(
        "Backward window function",
        6,
        FRAME_ECHO_WINDOW_LINEAR,
        kWindowFunctionPopup,
        0,
        FRAME_ECHO_BACKWARD_WINDOW_FUNCTION);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Forward window minimum",
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        PF_Precision_THOUSANDTHS,
        0,
        PF_ParamFlag_SUPERVISE,
        FRAME_ECHO_FORWARD_WINDOW_FALLOFF_RATIO);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Backward window minimum",
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        PF_Precision_THOUSANDTHS,
        0,
        0,
        FRAME_ECHO_BACKWARD_WINDOW_FALLOFF_RATIO);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUPX(
        "Forward window mapping",
        2,
        FRAME_ECHO_WINDOW_FALLOFF_OUTPUT_MAPPED,
        kWindowFalloffMappingPopup,
        PF_ParamFlag_SUPERVISE,
        FRAME_ECHO_FORWARD_WINDOW_FALLOFF_MAPPING);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUPX(
        "Backward window mapping",
        2,
        FRAME_ECHO_WINDOW_FALLOFF_OUTPUT_MAPPED,
        kWindowFalloffMappingPopup,
        0,
        FRAME_ECHO_BACKWARD_WINDOW_FALLOFF_MAPPING);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUPX(
        "Frame shortage behavior",
        3,
        FRAME_ECHO_FRAME_SHORTAGE_BLANK_TRANSPARENT,
        kFrameShortagePopup,
        0,
        FRAME_ECHO_FRAME_SHORTAGE_BEHAVIOR);

    AEFX_CLR_STRUCT(def);
    PF_ADD_COLOR(
        "Frame shortage color",
        0,
        0,
        0,
        FRAME_ECHO_FRAME_SHORTAGE_COLOR);

    out_data->num_params = FRAME_ECHO_NUM_PARAMS;

    return PF_Err_NONE;
}

static PF_Err UpdateParamsUI(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    return RefreshParamUI(in_data, out_data, params);
}

static PF_Err UserChangedParam(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output,
    const PF_UserChangedParamExtra* extra)
{
    const PF_ParamIndex changedIndex = extra ? extra->param_index : PF_ParamIndex_NONE;
    switch (changedIndex)
    {
    case FRAME_ECHO_SYNC_FORWARD_BACKWARD:
    case FRAME_ECHO_FORWARD_MAX_OPACITY:
    case FRAME_ECHO_FORWARD_WINDOW_FUNCTION:
    case FRAME_ECHO_FORWARD_WINDOW_FALLOFF_RATIO:
    case FRAME_ECHO_FORWARD_WINDOW_FALLOFF_MAPPING:
        ApplySyncValues(params);
        break;
    default:
        break;
    }

    // Force a re-render so AE actually calls PF_Cmd_RENDER after this param change.
    // Without this flag, AE may skip the render even when change_flags has CHANGED_VALUE set.
    out_data->out_flags |= PF_OutFlag_FORCE_RERENDER;

    return RefreshParamUI(in_data, out_data, params);
}

static frame_echo::PixelRGBA ConvertVUYAtoRGBA(float v, float u, float y, float a)
{
    frame_echo::PixelRGBA pixel;
    pixel.r = (y * 1.0f) + (u * 0.0f) + (v * 1.402f);
    pixel.g = (y * 1.0f) + (u * -0.344136f) + (v * -0.714136f);
    pixel.b = (y * 1.0f) + (u * 1.772f) + (v * 0.0f);
    pixel.a = a;
    return pixel;
}

static void ConvertRGBAToVUYA(const frame_echo::PixelRGBA& pixel, float& v, float& u, float& y, float& a)
{
    const float r = pixel.r;
    const float g = pixel.g;
    const float b = pixel.b;
    y = (r * 0.299f) + (g * 0.587f) + (b * 0.114f);
    u = (r * -0.168736f) + (g * -0.331264f) + (b * 0.5f);
    v = (r * 0.5f) + (g * -0.418688f) + (b * -0.081312f);
    a = pixel.a;
}

static frame_echo::PixelRGBA ReadPixelRGBA(
    const PF_InData* in_data,
    const PF_LayerDef* src,
    int x,
    int y,
    bool isVUYA)
{
    if (x < 0 || x >= src->width || y < 0 || y >= src->height)
    {
        return frame_echo::PixelRGBA{};
    }

    if (isPremiere)
    {
        // Premiere Pro: always float32 pixel data.
        // Each pixel is 4 floats, use rowbytes * height + x * 4 to calculate offset.
        const float* pixelBase = reinterpret_cast<const float*>(src->data);
        const size_t pixelOffset = static_cast<size_t>(y) * src->rowbytes / sizeof(float) + static_cast<size_t>(x) * 4;
        const float* pixel = pixelBase + pixelOffset;

        if (isVUYA)
        {
            // Pr VUYA_4444_32f: [V, U, Y, A]
            return ConvertVUYAtoRGBA(pixel[0], pixel[1], pixel[2], pixel[3]);
        }
        // Pr BGRA_4444_32f: [B, G, R, A]
        return frame_echo::PixelRGBA{
            pixel[2],  // R
            pixel[1],  // G
            pixel[0],  // B
            pixel[3],  // A
        };
    }

    // After Effects (8-bit, no Smart Render):
    // PF_Pixel {alpha, red, green, blue} as unsigned char.
    const PF_Pixel* pixelBase = reinterpret_cast<const PF_Pixel*>(src->data);
    const size_t pixelOffset = static_cast<size_t>(y) * src->rowbytes / sizeof(PF_Pixel) + static_cast<size_t>(x);
    const PF_Pixel& p = pixelBase[pixelOffset];
    constexpr float kInv255 = 1.0f / 255.0f;
    return frame_echo::PixelRGBA{
        static_cast<float>(p.red) * kInv255,
        static_cast<float>(p.green) * kInv255,
        static_cast<float>(p.blue) * kInv255,
        static_cast<float>(p.alpha) * kInv255,
    };
}

static void WritePixelRGBA(
    const PF_InData* in_data,
    PF_LayerDef* dest,
    int x,
    int y,
    const frame_echo::PixelRGBA& pixel,
    bool isVUYA)
{
    if (isPremiere)
    {
        // Premiere Pro: always float32 pixel data.
        // Each pixel is 4 floats, use rowbytes * height + x * 4 to calculate offset.
        float* pixelBase = reinterpret_cast<float*>(dest->data);
        const size_t pixelOffset = static_cast<size_t>(y) * dest->rowbytes / sizeof(float) + static_cast<size_t>(x) * 4;
        float* outPixel = pixelBase + pixelOffset;

        if (isVUYA)
        {
            float v, u, yVal, a;
            ConvertRGBAToVUYA(pixel, v, u, yVal, a);
            outPixel[0] = v;
            outPixel[1] = u;
            outPixel[2] = yVal;
            outPixel[3] = a;
        }
        else
        {
            // Pr BGRA_4444_32f: [B, G, R, A]
            outPixel[0] = pixel.b;
            outPixel[1] = pixel.g;
            outPixel[2] = pixel.r;
            outPixel[3] = pixel.a;
        }
        return;
    }

    // After Effects (8-bit, no Smart Render):
    // PF_Pixel {alpha, red, green, blue} as unsigned char.
    PF_Pixel* pixelBase = reinterpret_cast<PF_Pixel*>(dest->data);
    size_t pixelOffset = static_cast<size_t>(y) * dest->rowbytes / sizeof(PF_Pixel) + static_cast<size_t>(x);
    PF_Pixel& p = pixelBase[pixelOffset];
    constexpr float k255 = 255.0f;
    p.alpha = static_cast<A_u_char>(frame_echo::Clamp01(pixel.a) * k255 + 0.5f);
    p.red   = static_cast<A_u_char>(frame_echo::Clamp01(pixel.r) * k255 + 0.5f);
    p.green = static_cast<A_u_char>(frame_echo::Clamp01(pixel.g) * k255 + 0.5f);
    p.blue  = static_cast<A_u_char>(frame_echo::Clamp01(pixel.b) * k255 + 0.5f);
}

static frame_echo::PixelRGBA ReadColorParam(const PF_ParamDef* param)
{
    const float red = param->u.cd.value.red / 255.0f;
    const float green = param->u.cd.value.green / 255.0f;
    const float blue = param->u.cd.value.blue / 255.0f;
    return frame_echo::PixelRGBA{red, green, blue, 1.0f};
}

struct CheckedOutFrame
{
    PF_ParamDef param;
    const PF_LayerDef* layer = nullptr;
    bool checkedOut = false;
    bool valid = false;
};

struct SampleSource
{
    const PF_LayerDef* layer = nullptr;
    float weight = 1.0f;
    bool isCurrent = false;
};

struct RenderContext
{
    PF_InData* in_data = nullptr;
    PF_OutData* out_data = nullptr;
    PF_LayerDef* output = nullptr;
    const PF_LayerDef* src = nullptr;
    const std::vector<SampleSource>* sources = nullptr;
    frame_echo::BlendMode blendMode = frame_echo::BlendMode::BlendNewOnTop;
    float currentOpacity = 1.0f;
    bool isVUYA = false;
    frame_echo::FrameShortageBehavior shortageBehavior = frame_echo::FrameShortageBehavior::BlankTransparent;
    frame_echo::PixelRGBA shortageColor = {};
};

static bool IsTimeInRange(const PF_InData* in_data, A_long timeValue)
{
    if (in_data->total_time <= 0)
    {
        return timeValue >= 0;
    }
    return timeValue >= 0 && timeValue <= in_data->total_time;
}

static void CheckoutFrame(
    PF_InData* in_data,
    A_long timeValue,
    bool canCheckout,
    CheckedOutFrame& frame)
{
    AEFX_CLR_STRUCT(frame.param);
    if (!canCheckout || !IsTimeInRange(in_data, timeValue))
    {
        return;
    }

    PF_Err err = PF_Err_NONE;

    ERR(PF_CHECKOUT_PARAM(in_data, FRAME_ECHO_INPUT, timeValue, in_data->time_step, in_data->time_scale, &frame.param));
    if (err == PF_Err_NONE)
    {
        frame.checkedOut = true;
        if (frame.param.u.ld.data)
        {
            frame.layer = &frame.param.u.ld;
            frame.valid = true;
        }
    }
}

static void RenderRows(
    const RenderContext& context,
    int yStart,
    int yEnd,
    int xStart,
    int xEnd)
{
    std::vector<frame_echo::TemporalSample> samples;
    samples.reserve(context.sources->size());
    const frame_echo::PixelRGBA blankTransparent = frame_echo::MakeBlankTransparent();

    // Reusable vectors for SIMD batch processing
    std::vector<frame_echo::TemporalSample> samples1;
    samples1.reserve(context.sources->size());
    std::vector<frame_echo::TemporalSample> samples2;
    samples2.reserve(context.sources->size());
    std::vector<frame_echo::TemporalSample> samples3;
    samples3.reserve(context.sources->size());

    // Small helper lambda to populate one entry in all sample vectors from a source.
    // `p...` are the pixel values for pixel 0..N at the current source index.
    auto pushSamples = [&](const frame_echo::PixelRGBA& p0) {
        samples.push_back(frame_echo::TemporalSample{p0, 0.0f});
    };
    auto pushSamples2 = [&](const frame_echo::PixelRGBA& p0, const frame_echo::PixelRGBA& p1) {
        samples.push_back(frame_echo::TemporalSample{p0, 0.0f});
        samples1.push_back(frame_echo::TemporalSample{p1, 0.0f});
    };
    auto pushSamples4 = [&](const frame_echo::PixelRGBA& p0, const frame_echo::PixelRGBA& p1,
                             const frame_echo::PixelRGBA& p2, const frame_echo::PixelRGBA& p3) {
        samples.push_back(frame_echo::TemporalSample{p0, 0.0f});
        samples1.push_back(frame_echo::TemporalSample{p1, 0.0f});
        samples2.push_back(frame_echo::TemporalSample{p2, 0.0f});
        samples3.push_back(frame_echo::TemporalSample{p3, 0.0f});
    };

    // Read a pixel from a source layer (or apply shortage behavior).
    auto resolvePixel = [&](const SampleSource& source,
                             int px, int py,
                             const frame_echo::PixelRGBA& srcPx) -> frame_echo::PixelRGBA
    {
        if (source.isCurrent)
        {
            return srcPx;
        }
        if (source.layer)
        {
            return ReadPixelRGBA(context.in_data, source.layer, px, py, context.isVUYA);
        }
        if (context.shortageBehavior == frame_echo::FrameShortageBehavior::Repeat)
        {
            return srcPx;
        }
        if (context.shortageBehavior == frame_echo::FrameShortageBehavior::SolidColor)
        {
            return context.shortageColor;
        }
        return blankTransparent;
    };

    for (int y = yStart; y < yEnd; ++y)
    {
        int x = xStart;
        while (x < xEnd)
        {
            // --- AVX-512: process four pixels at once ---
            if (frame_echo::g_hasAVX512 && x + 3 < xEnd)
            {
                const frame_echo::PixelRGBA srcPx0 = ReadPixelRGBA(
                    context.in_data, context.src, x,     y, context.isVUYA);
                const frame_echo::PixelRGBA srcPx1 = ReadPixelRGBA(
                    context.in_data, context.src, x + 1, y, context.isVUYA);
                const frame_echo::PixelRGBA srcPx2 = ReadPixelRGBA(
                    context.in_data, context.src, x + 2, y, context.isVUYA);
                const frame_echo::PixelRGBA srcPx3 = ReadPixelRGBA(
                    context.in_data, context.src, x + 3, y, context.isVUYA);

                samples.clear();
                samples1.clear();
                samples2.clear();
                samples3.clear();
                for (const SampleSource& source : *context.sources)
                {
                    const frame_echo::PixelRGBA p0 = resolvePixel(source, x,     y, srcPx0);
                    const frame_echo::PixelRGBA p1 = resolvePixel(source, x + 1, y, srcPx1);
                    const frame_echo::PixelRGBA p2 = resolvePixel(source, x + 2, y, srcPx2);
                    const frame_echo::PixelRGBA p3 = resolvePixel(source, x + 3, y, srcPx3);

                    pushSamples4(p0, p1, p2, p3);
                }
                // Write back the correct weights after the loop
                for (std::size_t i = 0; i < samples.size(); ++i)
                {
                    samples[i].opacity = (*context.sources)[i].weight;
                    samples1[i].opacity = (*context.sources)[i].weight;
                    samples2[i].opacity = (*context.sources)[i].weight;
                    samples3[i].opacity = (*context.sources)[i].weight;
                }

                const frame_echo::PixelRGBA_4 quad = frame_echo::ComposeFourSamplesAVX512(
                    samples, samples1, samples2, samples3, context.blendMode);

                WritePixelRGBA(context.in_data, context.output, x,     y, quad.p0, context.isVUYA);
                WritePixelRGBA(context.in_data, context.output, x + 1, y, quad.p1, context.isVUYA);
                WritePixelRGBA(context.in_data, context.output, x + 2, y, quad.p2, context.isVUYA);
                WritePixelRGBA(context.in_data, context.output, x + 3, y, quad.p3, context.isVUYA);

                x += 4;
                continue;
            }

            // --- AVX2: process two pixels at once ---
            if (frame_echo::g_hasAVX2 && x + 1 < xEnd)
            {
                const frame_echo::PixelRGBA srcPx0 = ReadPixelRGBA(
                    context.in_data, context.src, x,     y, context.isVUYA);
                const frame_echo::PixelRGBA srcPx1 = ReadPixelRGBA(
                    context.in_data, context.src, x + 1, y, context.isVUYA);

                samples.clear();
                samples1.clear();
                for (const SampleSource& source : *context.sources)
                {
                    const frame_echo::PixelRGBA p0 = resolvePixel(source, x,     y, srcPx0);
                    const frame_echo::PixelRGBA p1 = resolvePixel(source, x + 1, y, srcPx1);

                    pushSamples2(p0, p1);
                }
                for (std::size_t i = 0; i < samples.size(); ++i)
                {
                    samples[i].opacity = (*context.sources)[i].weight;
                    samples1[i].opacity = (*context.sources)[i].weight;
                }

                const frame_echo::PixelRGBA_2 dual = frame_echo::ComposeTwoSamplesAVX2(
                    samples, samples1, context.blendMode);

                WritePixelRGBA(context.in_data, context.output, x,     y, dual.p0, context.isVUYA);
                WritePixelRGBA(context.in_data, context.output, x + 1, y, dual.p1, context.isVUYA);

                x += 2;
                continue;
            }

            // --- Single-pixel fallback ---
            const frame_echo::PixelRGBA srcPx = ReadPixelRGBA(
                context.in_data, context.src, x, y, context.isVUYA);

            samples.clear();
            for (const SampleSource& source : *context.sources)
            {
                const frame_echo::PixelRGBA pixel = resolvePixel(source, x, y, srcPx);
                pushSamples(pixel);
            }
            for (std::size_t i = 0; i < samples.size(); ++i)
            {
                samples[i].opacity = (*context.sources)[i].weight;
            }

            const frame_echo::PixelRGBA composed = frame_echo::ComposeSamples(samples, context.blendMode);
            WritePixelRGBA(context.in_data, context.output, x, y, composed, context.isVUYA);

            ++x;
        }
    }
}

static PF_Err Render(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_LayerDef* src = &params[FRAME_ECHO_INPUT]->u.ld;
    frame_echo::TemporalSettings settings = BuildTemporalSettings(params);
    settings.forwardMaxOpacity = frame_echo::Clamp01(settings.forwardMaxOpacity);
    settings.backwardMaxOpacity = frame_echo::Clamp01(settings.backwardMaxOpacity);

    bool isVUYA = false;
    if (isPremiere)
    {
        AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data);
        PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
        if (pixelFormatSuite->GetPixelFormat)
        {
            pixelFormatSuite->GetPixelFormat(output, &pixelFormat);
        }
        isVUYA = (pixelFormat == PrPixelFormat_VUYA_4444_32f) || (pixelFormat == PrPixelFormat_VUYA_4444_32f_709);
    }

    const bool canCheckout = (in_data->time_step != 0) && (in_data->time_scale != 0);
    const float currentOpacity = frame_echo::GetCurrentFrameOpacity(settings);
    const std::vector<float> backwardWeights = frame_echo::BuildBackwardWeights(
        settings.backwardFrameCount,
        settings.syncForwardBackward ? settings.forwardWindowFunction : settings.backwardWindowFunction,
        settings.syncForwardBackward ? settings.forwardWindowFalloffRatio : settings.backwardWindowFalloffRatio,
        settings.syncForwardBackward ? settings.forwardWindowFalloffMapping : settings.backwardWindowFalloffMapping,
        settings.syncForwardBackward ? settings.forwardMaxOpacity : settings.backwardMaxOpacity);
    const std::vector<float> forwardWeights = frame_echo::BuildForwardWeights(
        settings.forwardFrameCount,
        settings.forwardWindowFunction,
        settings.forwardWindowFalloffRatio,
        settings.forwardWindowFalloffMapping,
        settings.forwardMaxOpacity);

    std::vector<CheckedOutFrame> backwardFrames(backwardWeights.size());
    std::vector<CheckedOutFrame> forwardFrames(forwardWeights.size());

    for (std::size_t index = 0; index < backwardFrames.size(); ++index)
    {
        const A_long offset = static_cast<A_long>(index + 1) * in_data->time_step;
        CheckoutFrame(in_data, in_data->current_time - offset, canCheckout, backwardFrames[index]);
    }

    for (std::size_t index = 0; index < forwardFrames.size(); ++index)
    {
        const A_long offset = static_cast<A_long>(index + 1) * in_data->time_step;
        CheckoutFrame(in_data, in_data->current_time + offset, canCheckout, forwardFrames[index]);
    }

    std::vector<SampleSource> sources;
    sources.reserve(1 + backwardWeights.size() + forwardWeights.size());
    for (std::size_t index = 0; index < backwardWeights.size(); ++index)
    {
        sources.push_back(SampleSource{
            backwardFrames[index].valid ? backwardFrames[index].layer : nullptr,
            backwardWeights[index],
            false});
    }

    sources.push_back(SampleSource{src, currentOpacity, true});

    for (std::size_t index = 0; index < forwardWeights.size(); ++index)
    {
        sources.push_back(SampleSource{
            forwardFrames[index].valid ? forwardFrames[index].layer : nullptr,
            forwardWeights[index],
            false});
    }

    // Then we check whether all frames have same dimensions and pixel format as the current frame.
    // If not, we will treat those frames as unavailable and apply shortage behavior for them.
    // This is to prevent potential issues caused by incompatible frame data.
    for (SampleSource& source : sources)
    {
        if (source.layer)
        {
            if (source.layer->width != src->width ||
                source.layer->height != src->height ||
                source.layer->rowbytes != src->rowbytes)
            {
                source.layer = nullptr;
            }
        }
    }

    RenderContext context;
    context.in_data = in_data;
    context.out_data = out_data;
    context.output = output;
    context.src = src;
    context.sources = &sources;
    context.blendMode = settings.blendMode;
    context.currentOpacity = currentOpacity;
    context.isVUYA = isVUYA;
    context.shortageBehavior = settings.frameShortageBehavior;
    context.shortageColor = ReadColorParam(params[FRAME_ECHO_FRAME_SHORTAGE_COLOR]);

    // Only renders extent_hint
    RenderRows(
        context,
        std::max(0, output->extent_hint.top),
        std::min(output->height, output->extent_hint.bottom),
        std::max(0, output->extent_hint.left),
        std::min(output->width, output->extent_hint.right)
    );

    for (CheckedOutFrame& frame : backwardFrames)
    {
        if (frame.checkedOut)
        {
            PF_CHECKIN_PARAM(in_data, &frame.param);
        }
    }

    for (CheckedOutFrame& frame : forwardFrames)
    {
        if (frame.checkedOut)
        {
            PF_CHECKIN_PARAM(in_data, &frame.param);
        }
    }

    return PF_Err_NONE;
}

#ifdef MSWindows
#define DllExport __declspec(dllexport)
#else
#define DllExport __attribute__((visibility("default")))
#endif

extern "C" DllExport PF_Err EffectMain(
    PF_Cmd inCmd,
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* inOutput,
    void* extra)
{
    PF_Err err = PF_Err_NONE;

    switch (inCmd)
    {
    case PF_Cmd_GLOBAL_SETUP:
        err = GlobalSetup(in_data, out_data, params, inOutput);
        break;
    case PF_Cmd_GLOBAL_SETDOWN:
        err = GlobalSetdown(in_data, out_data, params, inOutput);
        break;
    case PF_Cmd_PARAMS_SETUP:
        err = ParamsSetup(in_data, out_data, params, inOutput);
        break;
    case PF_Cmd_UPDATE_PARAMS_UI:
        err = UpdateParamsUI(in_data, out_data, params, inOutput);
        break;
    case PF_Cmd_USER_CHANGED_PARAM:
        err = UserChangedParam(
            in_data,
            out_data,
            params,
            inOutput,
            reinterpret_cast<const PF_UserChangedParamExtra*>(extra));
        break;
    case PF_Cmd_RENDER:
        err = Render(in_data, out_data, params, inOutput);
        break;
    default:
        break;
    }

    return err;
}
