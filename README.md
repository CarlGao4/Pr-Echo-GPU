# Frame Echo (Pr-Echo-GPU)

## Introduction

Frame Echo is a GPU-accelerated temporal echo/blend plugin for **Adobe Premiere Pro** and **After Effects**. It creates motion trail and echo effects by blending multiple frames together with configurable weights, window functions, and blend modes.

## Features

- **Multi-frame temporal blending** - Blend up to 300 backward and 300 forward frames with the current frame
- **GPU acceleration** - Supports CUDA and OpenCL compute backends for real-time performance (with preserved DirectX backend which Pr and Ae will support in the near future)
- **CPU SIMD optimization** - Automatic runtime dispatch to SSE2, AVX2, or AVX-512 code paths
- **Multiple blend modes** - New On Top, New On Bottom, Add, Maximum, Minimum
- **Window functions** - Constant, Linear, Square Root, Square, Cosine, Smooth Step with configurable falloff
- **Per-direction opacity control** - Separate max opacity for forward and backward frames
- **Frame shortage handling** - Blank (transparent), Repeat, or Solid Color when frames are out of range
- **Dual host support** - Works as both a Premiere Pro GPU filter and an After Effects Smart Render effect

## System Requirements

- **Operating System**: Windows 10/11 (64-bit) (macOS support planned for future release)
- **Adobe Premiere Pro**: 2024 or later
- **Adobe After Effects**: 2024 release or later
- Hardware requirements:
  - **CUDA**: NVIDIA GPU with Compute Capability 5.0 or higher (e.g. GeForce GTX 750 or later)
  - **OpenCL**: Any GPU with OpenCL 3.0 support (e.g. Intel HD Graphics 4000 or later, AMD Radeon HD 7000 series or later)

## Installation

1. **Download the latest release** from the [GitHub Releases page](https://github.com/CarlGao4/Pr-Echo-GPU/releases/latest). You will need the `PrEchoGPU.aex` file.
2. **Find installation directory**: Usually the plugin should be saved to `C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore`. If the path could not be found, open `Regitry Editor` (`regedit.exe`) and navigate to `HKEY_LOCAL_MACHINE\SOFTWARE\Adobe\Premiere Pro\CurrentVersion` then check `Plug-InsDir` value for the correct path. If you didn't install Premiere Pro, you may also check `HKEY_LOCAL_MACHINE\SOFTWARE\Adobe\After Effects\[version]` and look for `CommonPluginInstallPath` value. Once you've located the correct plugin directory, both Premiere Pro and After Effects will be able to load the plugin.
3. **Copy the plugin**: Place the `PrEchoGPU.aex` file into the plugin directory you found in step 2. You may need administrator permissions to copy files into `Program Files`.
4. **Restart Adobe applications**: If Premiere Pro or After Effects were open during installation, restart them to load the new plugin.
5. **Verify installation**: Check the Effects panel under Video Effects > Time-PlugIn for "Frame Echo".

*Premiere Pro 2026 has a bug that some effects may not appear in the Effects panel though they are loaded. If you can't find it, please downgrade to Premiere Pro 2024 or 2025 until Adobe fixes the issue.*

## Parameters

| Parameter | Description |
|-----------|-------------|
| Backward Frames | Number of past frames to blend (0-300) |
| Forward Frames | Number of future frames to blend (0-300) |
| Sync Forward/Backward | Link forward and backward settings together |
| Blend Mode | How frames are combined (New On Top/Bottom, Add, Max, Min) |
| Forward/Backward Window Function | Weight distribution across frames (Constant, Linear, √, ², Cos, SmoothStep) |
| Window Falloff Ratio | Controls the shape of the window function curve |
| Window Falloff Mapping | Output-mapped or Input-mapped falloff |
| Forward/Backward Max Opacity | Maximum opacity for each direction |
| Frame Shortage Behavior | What to do when frames are out of range |
| Frame Shortage Color | Color to use when "Solid Color" shortage mode is selected |

## Build

This section is for developers who want to build the plugin from source.

### Build Requirements

- **OS**: Windows 10/11 (64-bit)
- **CMake** >= 3.24
- **MSVC** (Visual Studio 2019 or later, with C++17 support)
- **Adobe Premiere Pro SDK** 24.0 or later
- **After Effects SDK** 24.0 or later
- **Ninja** (optional)
- **CUDA Toolkit** >= 11.8 (optional, for CUDA backend)
- **OpenCL SDK** (optional, for OpenCL backend)
- **Boost** (optional, required for OpenCL kernel preprocessing)
- **Python 3** (optional, required for OpenCL/DirectX kernel embedding)

### Build Instructions

1. **Download and set up the SDKs**:
   - [Premiere Pro SDK](https://developer.adobe.com/console/755505/servicesandapis/pr)
     - You will need `Premiere Pro Plugin SDK`
     - Download version 2024 or later. Older versions may work but are not guaranteed.
   - [After Effects SDK](https://developer.adobe.com/console/755505/servicesandapis/ae)
     - You will need `After Effects Plug-in SDK`
     - Download version 2023 or later. Older versions may work but are not guaranteed.

2. **Clone the repository** and ensure the SDK folders are placed as siblings:

   ```
   Parent/
   ├── Pr-Echo-GPU/
   ├── Premiere_Pro_24.0_SDK/
   │   └── Examples/
   └── May2023_AfterEffectsSDK_Win/
       └── Examples/
   ```

   Alternatively, specify the SDK paths explicitly via CMake variables.

3. **Install dependencies**:
   - If you want to build with OpenCL support, make sure to install Boost and have the OpenCL SDK set up.
     - Download Boost from https://www.boost.org/releases/latest/
     - Download OpenCL SDK from https://github.com/KhronosGroup/OpenCL-SDK/releases
     - Install Python from https://www.python.org/downloads/
   - If you want to build with CUDA support, install the CUDA Toolkit from https://developer.nvidia.com/cuda-downloads

   - By default, all libraries above are detected automatically by CMake. If you have custom installation paths, add to `CMAKE_PREFIX_PATH` (using `-DCMAKE_PREFIX_PATH="path/to/boost;path/to/opencl;path/to/cuda"` when configuring with CMake).

4. **Configure with CMake**:

   ```bash
   cmake -B build -S . -G Ninja
   ```

   Optional CMake variables:

   | Variable | Default | Description |
   |----------|---------|-------------|
   | `GPU_ENABLED` | `ON` | Master switch for all GPU backends |
   | `GPU_ENABLE_CUDA` | `ON` | Enable CUDA backend |
   | `GPU_ENABLE_OPENCL` | `ON` | Enable OpenCL backend |
   | `GPU_ENABLE_DIRECTX` | `OFF` | Enable DirectX backend (though neither Pr nor Ae support it) |
   | `PREMIERE_SDK_ROOT` | Auto-detected | Path to Premiere Pro SDK `Examples` folder |
   | `AE_SDK_ROOT` | Auto-detected | Path to After Effects SDK `Examples` folder |
   | `PIPL_TOOL` | Auto-detected | Path to PiPL converter executable |
   | `CMAKE_CUDA_ARCHITECTURES` | `native` | Target CUDA architectures (e.g. `50;75;86;89`) |

   Example with CUDA disabled:

   ```bash
   cmake -B build -S . -G Ninja -DGPU_ENABLE_CUDA=OFF
   ```

   CPU-only build:

   ```bash
   cmake -B build -S . -G Ninja -DGPU_ENABLED=OFF
   ```

5. **Build**:

   ```bash
   cmake --build build
   ```

6. **Output**: The plugin `PrEchoGPU.aex` will be generated in the `build/` directory.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
