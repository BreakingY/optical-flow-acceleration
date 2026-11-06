cmake_minimum_required(VERSION 3.18)

project(grid_sample_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TRT_PATH "" CACHE PATH "TensorRT installation path")

if(NOT TRT_PATH)
    message(FATAL_ERROR
        "TRT_PATH is not set. "
        "Example: -DTRT_PATH=/path/to/TensorRT-8.4.1")
endif()

include_directories(
    ${TRT_PATH}/include
    /usr/local/cuda/include
)

link_directories(
    ${TRT_PATH}/lib
    /usr/local/cuda/lib64
)

add_library(grid_sample_plugin SHARED
    grid_sample_plugin.cpp
    grid_sample_plugin.cu
)

target_compile_features(
    grid_sample_plugin
    PRIVATE cxx_std_17
)

set_target_properties(
    grid_sample_plugin
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_link_libraries(
    grid_sample_plugin
    PRIVATE
        nvinfer
        cudart
)
