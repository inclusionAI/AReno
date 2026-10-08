cmake_minimum_required(VERSION 3.16)
project(areno_npu_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(RUN_MODE "npu" CACHE STRING "Ascend kernel execution mode")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
if(NOT SOC_VERSION OR NOT ARENO_ASCENDC_CMAKE_DIR)
    message(FATAL_ERROR "Build via setup.py to detect the CANN compiler and exact SoC")
endif()
message(STATUS "AReno Ascend C target: ${SOC_VERSION}")
include("${ARENO_ASCENDC_CMAKE_DIR}/ascendc.cmake")
# CANN includes its generated launch declarations before each source file.
# Kernel entrypoints must be global and expose only builtin template types;
# helper classes, enums and public host launchers may remain namespaced.
# Use template< (no space) at kernel definitions: CANN's legacy
# extract_src_template.py uses that spelling to enable template discovery.
# Entries called through host templates also have explicit device instances.
# Floating dtypes cross the generated host-stub boundary as integer IDs;
# half/BF16 names and their compiler-specific mangling stay inside kernels.
ascendc_library(areno_npu_kernels STATIC
    activation_kernel.cpp normalization_kernel.cpp optimizer_kernel.cpp
    optimizer_quantized_kernel.cpp optimizer_factored_kernel.cpp embedding_kernel.cpp
    linear_kernel.cpp conv_kernel.cpp attention_kernel.cpp routing_kernel.cpp moe_kernel.cpp
    fused_experts_kernel.cpp fused_experts_matmul_kernel.cpp)
