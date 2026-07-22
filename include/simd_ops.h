#ifndef TOKENIZER_SIMD_OPS_H
#define TOKENIZER_SIMD_OPS_H
/**
 * simd_ops.h — SIMD dispatch layer for bulk character classification
 *
 * Why: Classifying individual bytes (is this a letter? space? digit?) via
 * std::isalpha / std::isdigit on every byte is ~3-4 instructions per byte.
 * SIMD can classify 32 (AVX2) or 64 (AVX-512) bytes in a handful of
 * vector instructions — a ~10-20x throughput gain on the classification
 * bottleneck.
 *
 * Architecture:
 *   - Three kernel implementations (avx512, avx2, sse41/scalar) compiled
 *     as separate translation units with different /arch: flags.
 *   - A function pointer table resolved once at process startup via
 *     __cpuidex / _xgetbv. Hot path calls through the table — zero
 *     branches once dispatched.
 *
 * Character classes (bitmask design allows multiple classes per byte):
 *   BIT_LETTER  = 1   Unicode letter (Lu/Ll/Lt/Lm/Lo)
 *   BIT_DIGIT   = 2   Unicode number (Nd/Nl/No)
 *   BIT_SPACE   = 4   Whitespace (including tabs, NBSP, etc.)
 *   BIT_NEWLINE = 8   \r \n \u0085 \u2028 \u2029
 *   BIT_OTHER   = 16  Anything else (punctuation, symbols, etc.)
 *   BIT_UTF8_2  = 32  2-byte UTF-8 continuation starter
 *   BIT_UTF8_3  = 64  3-byte UTF-8 continuation starter
 *   BIT_UTF8_4  = 128 4-byte UTF-8 continuation starter
 */
#include <cstdint>
#include <cstddef>
#include <string_view>

// Character classification bit flags
enum : uint8_t {
    CLS_LETTER  = 1,
    CLS_DIGIT   = 2,
    CLS_SPACE   = 4,
    CLS_NEWLINE = 8,
    CLS_OTHER   = 16,
    CLS_UTF8_2  = 32,
    CLS_UTF8_3  = 64,
    CLS_UTF8_4  = 128,
};

struct ClassifyResult {
    uint8_t classes[64];     // class byte for each input byte (up to 64)
    uint32_t valid_utf8 : 1; // 0 = invalid UTF-8 sequence found
    uint32_t count      : 31;// number of bytes classified (≤ chunk_size)
};

struct WordBoundary {
    uint32_t start;  // byte offset into input
    uint32_t len;    // byte length
};

struct WordBoundariesResult {
    WordBoundary bounds[256]; // pre-allocated capacity (256 words per chunk)
    uint32_t count;
};

// CPU feature levels for dispatch
enum class SimdLevel : uint8_t {
    SCALAR = 0,
    SSE42  = 1,
    AVX2   = 2,
    AVX512 = 3,
};

// Function table — populated once at startup, never modified after
struct SimdOps {
    // Classify a chunk of bytes. Writes one class byte per input byte into out_classes.
    // Returns number of bytes classified.
    uint32_t (*classify)(const uint8_t* data, uint32_t len, ClassifyResult* out);

    // Find word boundaries in pre-classified bytes using GPT-2 rules
    // Returns number of word boundaries found
    uint32_t (*find_boundaries)(const uint8_t* data, const uint8_t* classes,
                                uint32_t len, WordBoundary* out_bounds,
                                uint32_t max_bounds);

    // Quick UTF-8 validity check on a span
    bool (*is_valid_utf8)(const uint8_t* data, size_t len);

    SimdLevel level;
    const char* name;
};

// The global dispatch table — initialized once
extern const SimdOps* g_simd;

// Initialize dispatch table (call once at startup)
// Detects CPU features, selects best available implementation
const SimdOps* init_simd_dispatch();

// Inline convenience wrappers
inline uint32_t simd_classify(const uint8_t* data, uint32_t len, ClassifyResult* out) {
    return g_simd->classify(data, len, out);
}

inline uint32_t simd_find_boundaries(const uint8_t* data, const uint8_t* classes,
                                     uint32_t len, WordBoundary* bounds, uint32_t max) {
    return g_simd->find_boundaries(data, classes, len, bounds, max);
}

inline bool simd_is_valid_utf8(const uint8_t* data, size_t len) {
    return g_simd->is_valid_utf8(data, len);
}

// Kernel function declarations (implemented in separate TUs)
uint32_t simd_classify_scalar(const uint8_t* data, uint32_t len, ClassifyResult* out);
uint32_t simd_classify_avx2(const uint8_t* data, uint32_t len, ClassifyResult* out);

uint32_t simd_boundaries_scalar(const uint8_t* data, const uint8_t* cls,
                                uint32_t len, WordBoundary* out, uint32_t max);
uint32_t simd_boundaries_avx2(const uint8_t* data, const uint8_t* cls,
                              uint32_t len, WordBoundary* out, uint32_t max);

bool simd_valid_utf8_scalar(const uint8_t* data, size_t len);
bool simd_valid_utf8_avx2(const uint8_t* data, size_t len);

#endif
