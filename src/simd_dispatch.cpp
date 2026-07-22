/**
 * simd_dispatch.cpp — CPU feature detection and dispatch table initialization
 *
 * Why: Resolving the SIMD backend once at process startup eliminates
 * branching on CPU features inside the hot loop. The dispatch table is
 * populated based on CPUID results and never modified afterward.
 *
 * Strategy:
 *   1. Check for AVX-512 (OS has enabled ZMM registers via XSAVE)
 *   2. Fall back to AVX2 (Haswell+, 2013)
 *   3. Fall back to SSE4.2 (Nehalem+, 2008)
 *   4. Fall back to portable scalar
 *
 * Each kernel is compiled as a separate translation unit with appropriate
 * /arch: flags. The linker picks the right symbols; the dispatcher selects
 * which function pointers to install in the table.
 */
#include "simd_ops.h"
#include <cstring>
#include <cstdio>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

static SimdOps g_ops;

static void cpu_id(int leaf, int* eax, int* ebx, int* ecx, int* edx) {
#ifdef _MSC_VER
    int info[4];
    __cpuidex(info, leaf, 0);
    *eax = info[0]; *ebx = info[1]; *ecx = info[2]; *edx = info[3];
#else
    __cpuid_count(leaf, 0, *eax, *ebx, *ecx, *edx);
#endif
}

static uint64_t xgetbv(uint32_t xcr) {
#ifdef _MSC_VER
    return _xgetbv(xcr);
#else
    uint32_t eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(xcr));
    return ((uint64_t)edx << 32) | eax;
#endif
}

const SimdOps* init_simd_dispatch() {
    int eax, ebx, ecx, edx;

    // Check max leaf
    cpu_id(0, &eax, &ebx, &ecx, &edx);
    int max_leaf = eax;

    if (max_leaf >= 1) {
        cpu_id(1, &eax, &ebx, &ecx, &edx);

        bool has_sse42 = (ecx >> 20) & 1;
        bool has_avx   = (ecx >> 28) & 1;
        bool has_xsave = (ecx >> 27) & 1;
        bool has_osxsave = (ecx >> 27) & 1;

        if (has_avx && has_xsave && has_osxsave && max_leaf >= 0x0D) {
            // Check OS has enabled AVX/AVX-512 registers
            uint64_t xcr0 = xgetbv(0);
            bool avx_supported  = (xcr0 & 0x06) == 0x06; // XMM+YMM
            bool avx512_supported = (xcr0 & 0xE6) == 0xE6; // XMM+YMM+ZMM+OPMASK

            if (avx512_supported && max_leaf >= 7) {
                cpu_id(7, &eax, &ebx, &ecx, &edx);
                bool has_avx512f  = (ebx >> 16) & 1;
                bool has_avx512bw = (ebx >> 30) & 1;
                bool has_avx512vl = (ebx >> 31) & 1;

                if (has_avx512f && has_avx512bw && has_avx512vl) {
                    g_ops.classify       = simd_classify_avx2; // AVX-512 not yet implemented
                    g_ops.find_boundaries= simd_boundaries_avx2;
                    g_ops.is_valid_utf8  = simd_valid_utf8_avx2;
                    g_ops.level          = SimdLevel::AVX512;
                    g_ops.name           = "AVX-512 (using AVX2 kernel)";
                    g_simd = &g_ops;
                    std::fprintf(stderr, "[tokenizer] SIMD: AVX-512\n");
                    return &g_ops;
                }
            }

            if (avx_supported) {
                cpu_id(7, &eax, &ebx, &ecx, &edx);
                bool has_avx2 = (ebx >> 5) & 1;
                if (has_avx2) {
                    g_ops.classify       = simd_classify_avx2;
                    g_ops.find_boundaries= simd_boundaries_avx2;
                    g_ops.is_valid_utf8  = simd_valid_utf8_avx2;
                    g_ops.level          = SimdLevel::AVX2;
                    g_ops.name           = "AVX2";
                    g_simd = &g_ops;
                    std::fprintf(stderr, "[tokenizer] SIMD: AVX2\n");
                    return &g_ops;
                }
            }

            if (has_sse42) {
                g_ops.classify       = simd_classify_scalar; // SSE4.2 not yet implemented
                g_ops.find_boundaries= simd_boundaries_scalar;
                g_ops.is_valid_utf8  = simd_valid_utf8_scalar;
                g_ops.level          = SimdLevel::SSE42;
                g_ops.name           = "SSE4.2";
                g_simd = &g_ops;
                std::fprintf(stderr, "[tokenizer] SIMD: SSE4.2\n");
                return &g_ops;
            }
        }
    }

    // Scalar fallback
    g_ops.classify       = simd_classify_scalar;
    g_ops.find_boundaries= simd_boundaries_scalar;
    g_ops.is_valid_utf8  = simd_valid_utf8_scalar;
    g_ops.level          = SimdLevel::SCALAR;
    g_ops.name           = "scalar";
    g_simd = &g_ops;
    std::fprintf(stderr, "[tokenizer] SIMD: scalar\n");
    return &g_ops;
}

const SimdOps* g_simd = nullptr;
