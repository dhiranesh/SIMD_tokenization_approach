cmake_minimum_required(VERSION 3.15)
project(qwen_tokenizer_v2
    VERSION 2.0.0
    DESCRIPTION "SIMD-accelerated Qwen3 BPE Tokenizer"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------------
# SIMD kernel compilation units — each compiled with different /arch:
# MSVC: /arch:AVX2 /arch:SSE2
# GCC/Clang: -mavx2 -mbmi2 (scalar uses default arch)
# ------------------------------------------------------------------
# AVX2 kernel object library
add_library(simd_avx2 OBJECT src/simd_avx2.cpp)
target_include_directories(simd_avx2 PRIVATE include)
if(MSVC)
    target_compile_options(simd_avx2 PRIVATE /arch:AVX2)
else()
    target_compile_options(simd_avx2 PRIVATE -mavx2 -mbmi2)
endif()

# Scalar kernel object library
add_library(simd_scalar OBJECT src/simd_scalar.cpp)
target_include_directories(simd_scalar PRIVATE include)
if(MSVC)
    # MSVC x64 default is SSE2
    target_compile_options(simd_scalar PRIVATE /arch:SSE2)
endif()

# ------------------------------------------------------------------
# Main library — SIMD kernels come from object libs (not compiled here)
# ------------------------------------------------------------------
add_library(tokenizer_v2_lib STATIC
    src/arena.cpp
    src/flat_vocab.cpp
    src/bpe_engine.cpp
    src/pre_tokenizer.cpp
    src/tokenizer_v2.cpp
    src/simd_dispatch.cpp
)

target_include_directories(tokenizer_v2_lib PUBLIC include)
target_link_libraries(tokenizer_v2_lib PRIVATE simd_avx2 simd_scalar)

if(MSVC)
    target_compile_options(tokenizer_v2_lib PRIVATE /utf-8 /W3 /O2)
else()
    target_compile_options(tokenizer_v2_lib PRIVATE -O3 -Wall -Wextra)
endif()

# ------------------------------------------------------------------
# Test executable
# ------------------------------------------------------------------
add_executable(qwen_tokenizer_v2
    main_v2.cpp
)

target_link_libraries(qwen_tokenizer_v2 PRIVATE tokenizer_v2_lib)
target_include_directories(qwen_tokenizer_v2 PRIVATE include)

# ------------------------------------------------------------------
# Enable address sanitizer in debug builds
# ------------------------------------------------------------------
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "Sanitize")
    if(MSVC)
        target_compile_options(tokenizer_v2_lib PRIVATE /fsanitize=address)
        target_link_options(tokenizer_v2_lib PRIVATE /fsanitize=address)
    else()
        target_compile_options(tokenizer_v2_lib PRIVATE -fsanitize=address -fno-omit-frame-pointer)
        target_link_options(tokenizer_v2_lib PRIVATE -fsanitize=address)
    endif()
endif()
