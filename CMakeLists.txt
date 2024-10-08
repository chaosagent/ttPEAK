cmake_minimum_required(VERSION 3.16)
project(ttpeak CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if("$ENV{TT_METAL_HOME}" STREQUAL "")
    message(FATAL_ERROR "TT_METAL_HOME is not set")
endif()
if("$ENV{ARCH_NAME}" STREQUAL "")
    message(FATAL_ERROR "ARCH_NAME is not set")
endif()

message($ENV{TT_METAL_HOME}/tt_metal/third_party/src/firmware/riscv/$ENV{ARCH_NAME})
add_executable(ttpeak ttpeak.cpp)
target_include_directories(ttpeak PRIVATE
    $ENV{TT_METAL_HOME}
    $ENV{TT_METAL_HOME}/tt_metal
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd
    $ENV{TT_METAL_HOME}/tt_metal/third_party/fmt
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc/$ENV{ARCH_NAME}
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc/
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/src/firmware/riscv/$ENV{ARCH_NAME}
)
if ($ENV{ARCH_NAME} MATCHES "wormhole_b0")
    target_include_directories(ttpeak PRIVATE
        $ENV{TT_METAL_HOME}/tt_metal/hw/inc/wormhole
        $ENV{TT_METAL_HOME}/tt_metal/hw/inc/wormhole/wormhole_b0_defines
        $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/src/firmware/riscv/wormhole
    )
endif()

if($ENV{ENABLE_PROFILER})
    message(STATUS "PROFILER ENABLED")
    target_compile_options(ttpeak PRIVATE -DPROFILER)
endif()

target_link_directories(ttpeak PRIVATE
    $ENV{TT_METAL_HOME}/build/tt_metal
)
target_link_libraries(ttpeak PRIVATE
    tt_metal
)

target_compile_definitions(ttpeak PRIVATE
    FMT_HEADER_ONLY
)

target_precompile_headers(ttpeak PRIVATE pch.hpp)
