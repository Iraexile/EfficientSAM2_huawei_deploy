# the minimum version of CMake.
cmake_minimum_required(VERSION 3.5.0)
project(EfficientSAM2)

set(NATIVERENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(DEFINED PACKAGE_FIND_FILE)
    include(${PACKAGE_FIND_FILE})
endif()

include_directories(${NATIVERENDER_ROOT_PATH}
                    ${NATIVERENDER_ROOT_PATH}/include)

add_library(entry SHARED napi_init.cpp mslite_runner.cpp)
target_link_libraries(entry PUBLIC libace_napi.z.so libmindspore_lite_ndk.so libhilog_ndk.z.so)

# Isolated GPU probe; existing CPU target is unchanged.
get_filename_component(SAM2_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(ncnn_DIR "${SAM2_ROOT}/dependencies/ncnn/ncnn-20261020-harmonyos-vulkan/${OHOS_ARCH}/lib/cmake/ncnn")
find_package(ncnn CONFIG REQUIRED)
add_library(gpu_probe SHARED gpu_bridge.cpp)
target_link_libraries(gpu_probe PRIVATE ncnn libace_napi.z.so libhilog_ndk.z.so)
