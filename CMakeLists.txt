cmake_minimum_required(VERSION 3.24)

project(PrEchoGPU LANGUAGES CXX RC)

if(NOT WIN32)
  message(FATAL_ERROR "PrEchoGPU only supports Windows.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type" FORCE)
endif()

# Use shared CUDA runtime to avoid LIBCMT conflict with /MD(d)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# ============================================================================
# GPU backend options - default ON; set to OFF to skip the whole backend
# ============================================================================
option(GPU_ENABLED        "Enable GPU backends"     ON)
option(GPU_ENABLE_OPENCL  "Enable OpenCL GPU backend"   ON)
option(GPU_ENABLE_DIRECTX "Enable DirectX GPU backend"  OFF)
option(GPU_ENABLE_CUDA    "Enable CUDA GPU backend"     ON)

if(NOT GPU_ENABLED)
  set(GPU_ENABLE_OPENCL OFF)
  set(GPU_ENABLE_DIRECTX OFF)
  set(GPU_ENABLE_CUDA OFF)
  message(STATUS "GPU_ENABLED is OFF, so all specific backends are disabled")
endif()

# ============================================================================
# CUDA Toolkit (>= 11.8) - static link the .cu kernel into the .aex
# ============================================================================
if(GPU_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit 11.8 REQUIRED)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native"
      CACHE STRING "CUDA architectures (default: native = your GPU only)")
    message(WARNING "CMAKE_CUDA_ARCHITECTURES is 'native' - this build only runs "
      "on your current GPU. For portable builds, set it explicitly:\n"
      "  cmake -DCMAKE_CUDA_ARCHITECTURES=\"50;75;86;89\" ..")
  endif()
endif()

# ============================================================================
# OpenCL - via find_package(CONFIG)
# ============================================================================
if(GPU_ENABLE_OPENCL)
  find_package(OpenCL CONFIG REQUIRED)
  get_target_property(_OCL_INCDIR OpenCL::Headers INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT _OCL_INCDIR)
    message(FATAL_ERROR "OpenCL::Headers target does not specify include directories")
  endif()
  message(STATUS "Found OpenCL: ${_OCL_INCDIR} (${OpenCL_VERSION})")
endif()

# ============================================================================
# DirectX Shader Compiler
# ============================================================================
if(GPU_ENABLE_DIRECTX)
  # We use powershell to get the dir because CMake can have trouble with the parentheses in "Program Files (x86)"
  execute_process(
    COMMAND powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('ProgramFiles(x86)')"
    OUTPUT_VARIABLE PROGRAM_FILES_X86_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  file(GLOB WINSDK_BIN_DIRS "${PROGRAM_FILES_X86_RAW}/Windows Kits/10/bin/*")
  set(DXC_FALLBACK_PATHS)
  foreach(dir IN LISTS WINSDK_BIN_DIRS)
      if(IS_DIRECTORY "${dir}/x64")
          list(APPEND DXC_FALLBACK_PATHS "${dir}/x64")
      endif()
  endforeach()
  find_program(DXC_EXECUTABLE dxc REQUIRED PATHS ${DXC_FALLBACK_PATHS})
  execute_process(COMMAND "${DXC_EXECUTABLE}" --version
    OUTPUT_VARIABLE DXC_VER ERROR_VARIABLE DXC_VER)
  string(STRIP "${DXC_VER}" DXC_VER)
  message(STATUS "DXC: ${DXC_EXECUTABLE} (${DXC_VER})")
endif()

# ============================================================================
# Boost (for OpenCL kernel preprocessing via MSVC /P)
# ============================================================================
if(GPU_ENABLE_OPENCL)
  find_package(Boost REQUIRED)
endif()

# Disable GPU_ENABLED if all specific backends are OFF
if(NOT GPU_ENABLE_OPENCL AND NOT GPU_ENABLE_DIRECTX AND NOT GPU_ENABLE_CUDA AND GPU_ENABLED)
  set(GPU_ENABLED OFF)
  message(STATUS "GPU_ENABLED set to OFF since all specific backends are disabled")
endif()

# ============================================================================
# Python3 (for CreateCString.py)
# ============================================================================
if((GPU_ENABLE_OPENCL OR GPU_ENABLE_DIRECTX))
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  message(STATUS "Python3: ${Python3_EXECUTABLE} (${Python3_VERSION})")
endif()

# ============================================================================
# Adobe SDK paths - no cmake config, manual search
# ============================================================================
set(PREMIERE_SDK_ROOT "" CACHE PATH "Premiere Pro SDK Examples root")
set(AE_SDK_ROOT       "" CACHE PATH "After Effects SDK Examples root")
set(PIPL_TOOL         "" CACHE FILEPATH "PiPL converter executable")

if(NOT PREMIERE_SDK_ROOT)
  get_filename_component(PREMIERE_SDK_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../Premiere_Pro_24.0_SDK/Examples" ABSOLUTE)
endif()
if(NOT AE_SDK_ROOT)
  get_filename_component(AE_SDK_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../May2023_AfterEffectsSDK_Win/Examples" ABSOLUTE)
endif()

if(NOT EXISTS "${PREMIERE_SDK_ROOT}")
  message(FATAL_ERROR "PREMIERE_SDK_ROOT not found. Set -DPREMIERE_SDK_ROOT=<path>")
endif()
if(NOT EXISTS "${AE_SDK_ROOT}")
  message(FATAL_ERROR "AE_SDK_ROOT not found. Set -DAE_SDK_ROOT=<path>")
endif()

if(NOT PIPL_TOOL)
  # When PIPL_TOOL is empty, find_program will be skipped, so we explicitly unset it from the cache
  unset(PIPL_TOOL CACHE)
  # We want to find PIPL_TOOL in Ae SDK first, so we don't set HINTS for Premiere SDK until the second find_program call
  find_program(PIPL_TOOL
    NAMES Cnvtpipl.exe Cnvtpipl PiPLTool.exe PiPLTool PiPLtool.exe PiPLtool
    HINTS "${AE_SDK_ROOT}/Resources/Win"
    HINTS "${AE_SDK_ROOT}/Resources")
endif()

if(NOT PIPL_TOOL)
  unset(PIPL_TOOL CACHE)
  find_program(PIPL_TOOL
    NAMES Cnvtpipl.exe Cnvtpipl PiPLTool.exe PiPLTool PiPLtool.exe PiPLtool
    HINTS "${PREMIERE_SDK_ROOT}/Resources/Win"
    HINTS "${PREMIERE_SDK_ROOT}/Resources")
endif()
if(NOT PIPL_TOOL)
  message(FATAL_ERROR "PIPL_TOOL not found. Set -DPIPL_TOOL=<path>")
endif()
message(STATUS "PIPL_TOOL: ${PIPL_TOOL}")

# ============================================================================
# SDK include directories
# ============================================================================
set(PREMIERE_HEADERS_DIR "${PREMIERE_SDK_ROOT}/Headers")
set(PREMIERE_UTILS_DIR  "${PREMIERE_SDK_ROOT}/Utils")
set(AE_HEADERS_DIR      "${AE_SDK_ROOT}/Headers")
set(AE_HEADERS_SP_DIR   "${AE_SDK_ROOT}/Headers/SP")
set(AE_RESOURCES_DIR    "${AE_SDK_ROOT}/Resources")
set(AE_UTILS_DIR        "${AE_SDK_ROOT}/Util")
set(GPUVIDEOFILTER_UTILS_DIR "${PREMIERE_SDK_ROOT}/Projects/GPUVideoFilter/Utils")

set(SDK_INCLUDE_DIRS
  "${PREMIERE_HEADERS_DIR}"  "${PREMIERE_UTILS_DIR}"
  "${AE_HEADERS_DIR}"        "${AE_HEADERS_SP_DIR}"
  "${AE_RESOURCES_DIR}"      "${AE_UTILS_DIR}"
  "${GPUVIDEOFILTER_UTILS_DIR}")

foreach(_d IN LISTS SDK_INCLUDE_DIRS)
  if(NOT EXISTS "${_d}")
    message(FATAL_ERROR "SDK directory not found: ${_d}")
  endif()
  message(STATUS "SDK include: ${_d}")
endforeach()

# ============================================================================
# Static library: PrEchoCore (CPU SIMD)
# ============================================================================
add_library(PrEchoCore STATIC
  src/FrameEchoCore.cpp
  src/FrameEchoCore_SIMD_SSE2.cpp
  src/FrameEchoCore_SIMD_AVX2.cpp
  src/FrameEchoCore_SIMD_AVX512.cpp)

target_include_directories(PrEchoCore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(PrEchoCore PRIVATE _USE_MATH_DEFINES)
set_source_files_properties(src/FrameEchoCore_SIMD_SSE2.cpp  PROPERTIES COMPILE_FLAGS "/arch:AVX2")
set_source_files_properties(src/FrameEchoCore_SIMD_AVX2.cpp  PROPERTIES COMPILE_FLAGS "/arch:AVX2")
set_source_files_properties(src/FrameEchoCore_SIMD_AVX512.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX512")

# ============================================================================
# PiPL / RC resource generation
# ============================================================================
set(GENERATED_ROOT "${CMAKE_CURRENT_BINARY_DIR}")
set(GENERATED_RC   "${GENERATED_ROOT}/FrameEcho.rc")
set(GENERATED_RCP  "${GENERATED_ROOT}/FrameEcho.rcp")
set(GENERATED_RR   "${GENERATED_ROOT}/FrameEcho.rr")

add_custom_command(
  OUTPUT "${GENERATED_RC}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/FrameEcho.rc" "${GENERATED_RC}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/FrameEcho.rc"
  VERBATIM)

if(GPU_ENABLED)
  set(RR_FLAGS "/D" "GPU_ENABLED=1")
else()
  set(RR_FLAGS "")
endif()

add_custom_command(
  OUTPUT "${GENERATED_RR}"
  COMMAND ${CMAKE_CXX_COMPILER} /nologo /D PRWIN_ENV
    /I "${AE_HEADERS_DIR}" /I "${AE_RESOURCES_DIR}" ${RR_FLAGS}
    /EP "${CMAKE_CURRENT_SOURCE_DIR}/FrameEcho.r" > "${GENERATED_RR}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/FrameEcho.r"
  VERBATIM)

add_custom_command(
  OUTPUT "${GENERATED_RCP}"
  COMMAND "${PIPL_TOOL}" "${GENERATED_RR}" "${GENERATED_RCP}"
  DEPENDS "${GENERATED_RR}"
  VERBATIM)

add_custom_target(PrEcho_Generated ALL
  DEPENDS "${GENERATED_RC}" "${GENERATED_RCP}")

# ============================================================================
# OpenCL kernel - preprocess (.i) then embed as C string (.cl.h)
# Two separate commands for easier debugging of the preprocessing step.
# ============================================================================
if(GPU_ENABLE_OPENCL)
  set(GENERATED_OPENCL_DIR    "${GENERATED_ROOT}/PreprocessedOpenCL")
  set(GENERATED_OPENCL_I      "${GENERATED_ROOT}/FrameEchoBlend.i")
  set(GENERATED_OPENCL_HEADER "${GENERATED_OPENCL_DIR}/FrameEchoBlend.cl.h")

  add_custom_command(
    OUTPUT ${GENERATED_OPENCL_I}
    COMMAND ${CMAKE_CXX_COMPILER} /nologo /D GF_DEVICE_TARGET_OPENCL=1
      /I${PREMIERE_HEADERS_DIR} /I${_OCL_INCDIR} /I${Boost_INCLUDE_DIRS}
      /TP /P /Fi${GENERATED_OPENCL_I}
      ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/FrameEchoBlend.cl
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/line_path_removal_inplace.py
      ${GENERATED_OPENCL_I}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/FrameEchoBlend.cl
    VERBATIM)

  add_custom_command(
    OUTPUT ${GENERATED_OPENCL_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_OPENCL_DIR}
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/bin2h.py
      -i ${GENERATED_OPENCL_I} -o ${GENERATED_OPENCL_HEADER}
      -n kFrameEchoBlend_OpenCLString
    DEPENDS ${GENERATED_OPENCL_I}
    VERBATIM)
endif()

# ============================================================================
# DirectX shader - compile .hlsl -> .cso / .rs, then embed as byte arrays
# via bin2h.py (in src/kernels/).
# ============================================================================
if(GPU_ENABLE_DIRECTX)
  set(GENERATED_DX_DIR    "${GENERATED_ROOT}/DirectX_Assets")
  set(GENERATED_DXC_CSO   "${GENERATED_DX_DIR}/FrameEchoBlend.cso")
  set(GENERATED_DXC_RS    "${GENERATED_DX_DIR}/FrameEchoBlend.rs")
  set(GENERATED_DX_CSO_HEADER "${GENERATED_DX_DIR}/FrameEchoBlend_cso.h")
  set(GENERATED_DX_RS_HEADER  "${GENERATED_DX_DIR}/FrameEchoBlend_rs.h")

  add_custom_command(
    OUTPUT ${GENERATED_DXC_CSO} ${GENERATED_DXC_RS}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_DX_DIR}
    COMMAND ${DXC_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/FrameEchoBlend.hlsl
      -E main -Fo ${GENERATED_DXC_CSO} -Frs ${GENERATED_DXC_RS}
      -T cs_6_5 -enable-16bit-types
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/FrameEchoBlend.hlsl
    VERBATIM)

  add_custom_command(
    OUTPUT ${GENERATED_DX_CSO_HEADER}
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/bin2h.py
      -i ${GENERATED_DXC_CSO} -o ${GENERATED_DX_CSO_HEADER}
      -n kFrameEchoBlend_CSO
    DEPENDS ${GENERATED_DXC_CSO}
    VERBATIM)

  add_custom_command(
    OUTPUT ${GENERATED_DX_RS_HEADER}
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/bin2h.py
      -i ${GENERATED_DXC_RS} -o ${GENERATED_DX_RS_HEADER}
      -n kFrameEchoBlend_RootSig
    DEPENDS ${GENERATED_DXC_RS}
    VERBATIM)
endif()

# ============================================================================
# Plugin shared library
# ============================================================================
set(PLUGIN_SOURCES src/FrameEchoEffect.cpp src/FrameEchoSmartRender.cpp)
# Then find Smart_Utils.cpp in SDK_INCLUDE_DIRS and add it
find_file(SMART_UTILS_CPP Smart_Utils.cpp PATHS ${SDK_INCLUDE_DIRS})
if(SMART_UTILS_CPP)
  list(APPEND PLUGIN_SOURCES ${SMART_UTILS_CPP})
else()
  message(FATAL_ERROR "Smart_Utils.cpp not found in SDK include directories")
endif()

if(GPU_ENABLED)
  list(APPEND PLUGIN_SOURCES src/FrameEchoPrGpu.cpp)
  list(APPEND PLUGIN_SOURCES src/FrameEchoAeGpu.cpp)
endif()
if(GPU_ENABLE_CUDA)
  list(APPEND PLUGIN_SOURCES src/kernels/FrameEchoBlend.cu)
endif()
if(GPU_ENABLE_OPENCL)
  list(APPEND PLUGIN_SOURCES "${GENERATED_OPENCL_HEADER}")
endif()
if(GPU_ENABLE_DIRECTX)
  list(APPEND PLUGIN_SOURCES "${GENERATED_DX_CSO_HEADER}" "${GENERATED_DX_RS_HEADER}")
endif()

add_library(PrEchoPlugin SHARED ${PLUGIN_SOURCES})

add_dependencies(PrEchoPlugin PrEcho_Generated)

target_include_directories(PrEchoPlugin PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
  ${SDK_INCLUDE_DIRS}
  ${GENERATED_ROOT})

target_compile_definitions(PrEchoPlugin PRIVATE
  MSWindows PRWIN_ENV _WINDOWS _USE_MATH_DEFINES PRSDK_GPU_SUPPORT)

if(GPU_ENABLED)
  target_compile_definitions(PrEchoPlugin PRIVATE GPU_ENABLED=1)
endif()

# ---- Per-backend ----
if(GPU_ENABLE_CUDA)
  target_compile_definitions(PrEchoPlugin PRIVATE GPU_ENABLE_CUDA=1)
  target_link_libraries(PrEchoPlugin PRIVATE CUDA::cuda_driver)
endif()

if(GPU_ENABLE_OPENCL)
  target_compile_definitions(PrEchoPlugin PRIVATE
    GPU_ENABLE_OPENCL=1
    CL_TARGET_OPENCL_VERSION=300)
  target_include_directories(PrEchoPlugin PRIVATE
    "${_OCL_INCDIR}" "${GENERATED_OPENCL_DIR}")
  target_link_libraries(PrEchoPlugin PRIVATE OpenCL::OpenCL)
endif()

if(GPU_ENABLE_DIRECTX)
  target_compile_definitions(PrEchoPlugin PRIVATE GPU_ENABLE_DIRECTX=1)
  target_include_directories(PrEchoPlugin PRIVATE "${GENERATED_DX_DIR}")
  target_link_libraries(PrEchoPlugin PRIVATE d3dcompiler d3d12 dxgi)
  target_sources(PrEchoPlugin PRIVATE "${GPUVIDEOFILTER_UTILS_DIR}/DirectXUtils.cpp")
endif()

target_link_libraries(PrEchoPlugin PRIVATE
  PrEchoCore vfw32 msacm32 winmm comctl32 uuid)

target_sources(PrEchoPlugin PRIVATE "${GENERATED_RC}")

set_target_properties(PrEchoPlugin PROPERTIES
  OUTPUT_NAME "PrEchoGPU"
  PREFIX ""
  SUFFIX ".aex")
