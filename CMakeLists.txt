cmake_minimum_required(VERSION 3.20)

project(PrEchoGPU LANGUAGES CXX RC)

if(NOT WIN32)
  message(FATAL_ERROR "PrEchoGPU only supports Windows.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(DEFAULT_PREMIERE_SDK_ROOT "${CMAKE_CURRENT_LIST_DIR}/../Premiere_Pro_24.0_SDK/Examples" ABSOLUTE)
get_filename_component(DEFAULT_AE_SDK_ROOT "${CMAKE_CURRENT_LIST_DIR}/../May2023_AfterEffectsSDK_Win/Examples" ABSOLUTE)

set(PREMIERE_SDK_ROOT "${DEFAULT_PREMIERE_SDK_ROOT}" CACHE PATH "Premiere Pro SDK root")
set(AE_SDK_ROOT "${DEFAULT_AE_SDK_ROOT}" CACHE PATH "After Effects SDK root")
set(PIPL_TOOL "" CACHE FILEPATH "PiPL converter executable")

if(NOT EXISTS "${PREMIERE_SDK_ROOT}")
  message(FATAL_ERROR "PREMIERE_SDK_ROOT does not exist: ${PREMIERE_SDK_ROOT}")
endif()

if(NOT EXISTS "${AE_SDK_ROOT}")
  message(FATAL_ERROR "AE_SDK_ROOT does not exist: ${AE_SDK_ROOT}")
endif()

if(NOT PIPL_TOOL)
  unset(PIPL_TOOL CACHE)
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
  message(FATAL_ERROR "PIPL_TOOL was not found. Set PIPL_TOOL or install the AE SDK resources.")
endif()

message(STATUS "Using PiPL tool: ${PIPL_TOOL}")

add_library(PrEchoCore STATIC
  src/FrameEchoCore.cpp
  src/FrameEchoCore_SIMD_SSE2.cpp
  src/FrameEchoCore_SIMD_AVX2.cpp
  src/FrameEchoCore_SIMD_AVX512.cpp)

target_include_directories(PrEchoCore PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_definitions(PrEchoCore PRIVATE
  _USE_MATH_DEFINES)

# FrameEchoCore_SIMD_SSE2.cpp must be compiled with /arch:AVX2 so that
# all SSE/AVX intrinsics emit VEX-coded instructions and FMA is available.
set_source_files_properties(src/FrameEchoCore_SIMD_SSE2.cpp PROPERTIES
  COMPILE_FLAGS "/arch:AVX2")

# FrameEchoCore_SIMD_AVX2.cpp uses true 256-bit AVX2 intrinsics.
set_source_files_properties(src/FrameEchoCore_SIMD_AVX2.cpp PROPERTIES
  COMPILE_FLAGS "/arch:AVX2")

# FrameEchoCore_SIMD_AVX512.cpp uses true 512-bit AVX-512 intrinsics.
set_source_files_properties(src/FrameEchoCore_SIMD_AVX512.cpp PROPERTIES
  COMPILE_FLAGS "/arch:AVX512")

add_library(PrEchoPlugin SHARED
  src/FrameEchoEffect.cpp
  # src/FrameEchoPrGpu.cpp  # Temporarily disabled to test if Pr detects the plugin via AE Effect path only
)

set(PREMIERE_HEADERS_DIR "${PREMIERE_SDK_ROOT}/Headers")
set(PREMIERE_UTILS_DIR "${PREMIERE_SDK_ROOT}/Utils")
set(AE_HEADERS_DIR "${AE_SDK_ROOT}/Headers")
set(AE_HEADERS_SP_DIR "${AE_SDK_ROOT}/Headers/SP")
set(AE_RESOURCES_DIR "${AE_SDK_ROOT}/Resources")
set(AE_UTILS_DIR "${AE_SDK_ROOT}/Util")

set(SDK_INCLUDE_DIRS
  "${PREMIERE_HEADERS_DIR}"
  "${PREMIERE_UTILS_DIR}"
  "${AE_HEADERS_DIR}"
  "${AE_HEADERS_SP_DIR}"
  "${AE_RESOURCES_DIR}"
  "${AE_UTILS_DIR}")

foreach(_sdk_dir IN LISTS SDK_INCLUDE_DIRS)
  if(NOT EXISTS "${_sdk_dir}")
    message(FATAL_ERROR "Required SDK directory not found: ${_sdk_dir}")
  endif()
endforeach()

message(STATUS "Using Premiere SDK root: ${PREMIERE_SDK_ROOT}")
message(STATUS "Using After Effects SDK root: ${AE_SDK_ROOT}")
foreach(_sdk_dir IN LISTS SDK_INCLUDE_DIRS)
  message(STATUS "Using SDK include dir: ${_sdk_dir}")
endforeach()

set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(GENERATED_ROOT "${CMAKE_CURRENT_BINARY_DIR}")
set(GENERATED_RC "${GENERATED_ROOT}/FrameEcho.rc")
set(GENERATED_RCP "${GENERATED_ROOT}/FrameEcho.rcp")
set(GENERATED_RR "${GENERATED_ROOT}/FrameEcho.rr")

add_custom_command(
  OUTPUT "${GENERATED_RC}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${GENERATED_ROOT}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PROJECT_ROOT}/FrameEcho.rc" "${GENERATED_RC}"
  DEPENDS "${PROJECT_ROOT}/FrameEcho.rc"
  VERBATIM)

add_custom_command(
  OUTPUT "${GENERATED_RR}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${GENERATED_ROOT}"
  COMMAND ${CMAKE_CXX_COMPILER} /nologo /D PRWIN_ENV /I "${AE_HEADERS_DIR}" /I "${AE_RESOURCES_DIR}" /EP "${PROJECT_ROOT}/FrameEcho.r" > "${GENERATED_RR}"
  DEPENDS "${PROJECT_ROOT}/FrameEcho.r"
  VERBATIM)

add_custom_command(
  OUTPUT "${GENERATED_RCP}"
  COMMAND "${PIPL_TOOL}" "${GENERATED_RR}" "${GENERATED_RCP}"
  DEPENDS "${GENERATED_RR}"
  VERBATIM)

add_custom_target(PrEcho_Generated ALL
  DEPENDS "${GENERATED_RC}" "${GENERATED_RCP}")

add_dependencies(PrEchoPlugin PrEcho_Generated)

target_include_directories(PrEchoPlugin PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${SDK_INCLUDE_DIRS})

target_compile_definitions(PrEchoPlugin PRIVATE
  MSWindows
  PRWIN_ENV
  _WINDOWS
  _USE_MATH_DEFINES)

target_link_libraries(PrEchoPlugin PRIVATE
  PrEchoCore
  vfw32
  msacm32
  winmm
  comctl32
  uuid)

set_target_properties(PrEchoPlugin PROPERTIES
  OUTPUT_NAME "PrEchoGPU"
  PREFIX ""
  SUFFIX ".aex")

target_sources(PrEchoPlugin PRIVATE "${GENERATED_RC}")
