/**
 * simd_avx2.cpp — AVX2-accelerated character classification
 *
 * Compiled separately with -mavx2 (GCC/Clang) or /arch:AVX2 (MSVC).
 * Falls back to scalar if AVX2 not available at compile time.
 */
#include "simd_ops.h"
#include <cstring>

#ifdef __AVX2__
#include <immintrin.h>
#endif

static bool is_letter_cp(uint32_t cp) {
    if (cp < 0x00C0) return false;
    if (cp >= 0x00C0 && cp <= 0x024F) return true;
    if (cp >= 0x0300 && cp <= 0x036F) return false;
    if (cp >= 0x0370 && cp <= 0x03FF) return true;
    if (cp >= 0x0400 && cp <= 0x04FF) return true;
    if (cp >= 0x0500 && cp <= 0x052F) return true;
    if (cp >= 0x0530 && cp <= 0x058F) return true;
    if (cp >= 0x0590 && cp <= 0x05FF) return true;
    if (cp >= 0x0600 && cp <= 0x06FF) return true;
    if (cp >= 0x0700 && cp <= 0x074F) return true;
    if (cp >= 0x0900 && cp <= 0x097F) return true;
    if (cp >= 0x0980 && cp <= 0x09FF) return true;
    if (cp >= 0x0A00 && cp <= 0x0A7F) return true;
    if (cp >= 0x0A80 && cp <= 0x0AFF) return true;
    if (cp >= 0x0B00 && cp <= 0x0B7F) return true;
    if (cp >= 0x0B80 && cp <= 0x0BFF) return true;
    if (cp >= 0x0C00 && cp <= 0x0C7F) return true;
    if (cp >= 0x0C80 && cp <= 0x0CFF) return true;
    if (cp >= 0x0D00 && cp <= 0x0D7F) return true;
    if (cp >= 0x0D80 && cp <= 0x0DFF) return true;
    if (cp >= 0x0E00 && cp <= 0x0E7F) return true;
    if (cp >= 0x0E80 && cp <= 0x0EFF) return true;
    if (cp >= 0x0F00 && cp <= 0x0FFF) return true;
    if (cp >= 0x1000 && cp <= 0x109F) return true;
    if (cp >= 0x10A0 && cp <= 0x10FF) return true;
    if (cp >= 0x1100 && cp <= 0x11FF) return true;
    if (cp >= 0x2E80 && cp <= 0x2EFF) return true;
    if (cp >= 0x2F00 && cp <= 0x2FDF) return true;
    if (cp >= 0x3040 && cp <= 0x309F) return true;
    if (cp >= 0x30A0 && cp <= 0x30FF) return true;
    if (cp >= 0x3100 && cp <= 0x312F) return true;
    if (cp >= 0x3130 && cp <= 0x318F) return true;
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
    if (cp >= 0xA000 && cp <= 0xA4CF) return true;
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;
    if (cp >= 0x20000 && cp <= 0x2A6DF) return true;
    if (cp >= 0x2A700 && cp <= 0x2B73F) return true;
    if (cp >= 0x2B740 && cp <= 0x2B81F) return true;
    if (cp >= 0x2B820 && cp <= 0x2CEAF) return true;
    if (cp >= 0x2F800 && cp <= 0x2FA1F) return true;
    if (cp >= 0x30000 && cp <= 0x3134F) return true;
    if (cp >= 0x31350 && cp <= 0x323AF) return true;
    if (cp >= 0x2100 && cp <= 0x214F) return true;
    if (cp >= 0x2C00 && cp <= 0x2C5F) return true;
    if (cp >= 0xA640 && cp <= 0xA69F) return true;
    if (cp >= 0x1D400 && cp <= 0x1D7FF) return true;
    return false;
}

static bool is_digit_cp(uint32_t cp) {
    if (cp >= 0x0660 && cp <= 0x0669) return true;
    if (cp >= 0x06F0 && cp <= 0x06F9) return true;
    if (cp >= 0x0966 && cp <= 0x096F) return true;
    if (cp >= 0x09E6 && cp <= 0x09EF) return true;
    if (cp >= 0x0BE6 && cp <= 0x0BEF) return true;
    if (cp >= 0x0E50 && cp <= 0x0E59) return true;
    if (cp >= 0x0ED0 && cp <= 0x0ED9) return true;
    if (cp >= 0x0F20 && cp <= 0x0F29) return true;
    if (cp >= 0x1040 && cp <= 0x1049) return true;
    if (cp >= 0x17E0 && cp <= 0x17E9) return true;
    if (cp >= 0x1810 && cp <= 0x1819) return true;
    if (cp >= 0x1B50 && cp <= 0x1B59) return true;
    if (cp >= 0xFF10 && cp <= 0xFF19) return true;
    if (cp >= 0x2070 && cp <= 0x2079) return true;
    if (cp >= 0x2080 && cp <= 0x2089) return true;
    return false;
}

static void propagate_utf8_classes(uint8_t* out_classes, const uint8_t* data,
                                    uint32_t process_len) {
    // Walk forward through UTF-8 bytes, decode each multi-byte sequence,
    // and set the correct class for all bytes based on the codepoint's
    // Unicode property (letter, digit, or other).
    // The SIMD path classifies each byte independently and doesn't handle
    // multi-byte chars, so we correct its output here.
    for (uint32_t j = 0; j < process_len; ) {
        uint8_t b = data[j];
        if (b <= 0x7F) {
            ++j; // ASCII — already correctly classified
        } else if ((b & 0xE0) == 0xC0) {
            if (j + 1 >= process_len) { out_classes[j] = CLS_OTHER; ++j; continue; }
            uint32_t cp = (b & 0x1F) << 6 | (data[j+1] & 0x3F);
            uint8_t cls = is_digit_cp(cp) ? CLS_DIGIT :
                          is_letter_cp(cp) ? CLS_LETTER : CLS_OTHER;
            out_classes[j] = cls;
            out_classes[j+1] = cls;
            j += 2;
        } else if ((b & 0xF0) == 0xE0) {
            if (j + 2 >= process_len) { out_classes[j] = CLS_OTHER; ++j; continue; }
            uint32_t cp = (b & 0x0F) << 12 | (data[j+1] & 0x3F) << 6 | (data[j+2] & 0x3F);
            uint8_t cls = is_digit_cp(cp) ? CLS_DIGIT :
                          is_letter_cp(cp) ? CLS_LETTER : CLS_OTHER;
            for (uint32_t k = 0; k < 3 && j + k < process_len; ++k)
                out_classes[j + k] = cls;
            j += 3;
        } else if ((b & 0xF8) == 0xF0) {
            if (j + 3 >= process_len) { out_classes[j] = CLS_OTHER; ++j; continue; }
            uint32_t cp = (b & 0x07) << 18 | (data[j+1] & 0x3F) << 12 |
                          (data[j+2] & 0x3F) << 6 | (data[j+3] & 0x3F);
            uint8_t cls = is_letter_cp(cp) ? CLS_LETTER : CLS_OTHER;
            for (uint32_t k = 0; k < 4 && j + k < process_len; ++k)
                out_classes[j + k] = cls;
            j += 4;
        } else {
            out_classes[j] = CLS_OTHER;
            ++j;
        }
    }
}

uint32_t simd_classify_avx2(const uint8_t* data, uint32_t len, ClassifyResult* out) {
#ifdef __AVX2__
    out->valid_utf8 = 1;
    uint32_t i = 0;

    // Process 32 bytes at a time (cap at 64 to match ClassifyResult::classes[])
    uint32_t process_len = (len > 64) ? 64 : len;
    uint8_t* out_classes = out->classes;

    for (; i + 32 <= process_len; i += 32) {
        __m256i chunk = _mm256_loadu_si256((const __m256i*)(data + i));

        __m256i bias = _mm256_set1_epi8((char)-128);
        __m256i s_chunk = _mm256_add_epi8(chunk, bias);

        // cmpgt is STRICT greater-than: byte > bound. So bounds are lo-1 and hi+1.
        __m256i lo_a = _mm256_set1_epi8((char)(0x40 - 128)); // 0x41-1
        __m256i hi_z = _mm256_set1_epi8((char)(0x5B - 128)); // 0x5A+1
        __m256i is_upper = _mm256_and_si256(
            _mm256_cmpgt_epi8(s_chunk, lo_a),
            _mm256_cmpgt_epi8(hi_z, s_chunk));

        __m256i lo_a2 = _mm256_set1_epi8((char)(0x60 - 128)); // 0x61-1
        __m256i hi_z2 = _mm256_set1_epi8((char)(0x7B - 128)); // 0x7A+1
        __m256i is_lower = _mm256_and_si256(
            _mm256_cmpgt_epi8(s_chunk, lo_a2),
            _mm256_cmpgt_epi8(hi_z2, s_chunk));

        __m256i is_letter = _mm256_or_si256(is_upper, is_lower);

        __m256i lo_0 = _mm256_set1_epi8((char)(0x2F - 128)); // 0x30-1
        __m256i hi_9 = _mm256_set1_epi8((char)(0x3A - 128)); // 0x39+1
        __m256i is_digit = _mm256_and_si256(
            _mm256_cmpgt_epi8(s_chunk, lo_0),
            _mm256_cmpgt_epi8(hi_9, s_chunk));

        __m256i nl1 = _mm256_set1_epi8('\n');
        __m256i nl2 = _mm256_set1_epi8('\r');
        __m256i is_nl = _mm256_or_si256(
            _mm256_cmpeq_epi8(chunk, nl1),
            _mm256_cmpeq_epi8(chunk, nl2));

        __m256i space_c = _mm256_set1_epi8(' ');
        __m256i tab_c  = _mm256_set1_epi8('\t');
        __m256i is_space = _mm256_or_si256(
            _mm256_cmpeq_epi8(chunk, space_c),
            _mm256_cmpeq_epi8(chunk, tab_c));

        __m256i letter_mask = _mm256_set1_epi8(CLS_LETTER);
        __m256i digit_mask  = _mm256_set1_epi8(CLS_DIGIT);
        __m256i space_mask  = _mm256_set1_epi8(CLS_SPACE);
        __m256i nl_mask     = _mm256_set1_epi8(CLS_NEWLINE);

        __m256i high_bit  = _mm256_set1_epi8((char)0x80);
        __m256i not_high  = _mm256_xor_si256(high_bit, _mm256_set1_epi8(0xFF));
        __m256i is_high   = _mm256_and_si256(chunk, high_bit);

        __m256i cont_mask = _mm256_set1_epi8((char)0xC0);
        __m256i is_cont   = _mm256_cmpeq_epi8(
            _mm256_and_si256(chunk, cont_mask),
            _mm256_set1_epi8((char)0x80));

        __m256i valid_range = _mm256_andnot_si256(is_cont, not_high);

        __m256i is_letter_v = _mm256_and_si256(is_letter, valid_range);
        __m256i is_digit_v  = _mm256_and_si256(is_digit, valid_range);
        __m256i is_space_v  = _mm256_and_si256(is_space, valid_range);
        __m256i is_nl_v     = _mm256_and_si256(is_nl, valid_range);

        __m256i result = _mm256_setzero_si256();
        result = _mm256_or_si256(result, _mm256_and_si256(is_letter_v, letter_mask));
        result = _mm256_or_si256(result, _mm256_and_si256(is_digit_v, digit_mask));
        result = _mm256_or_si256(result, _mm256_and_si256(is_space_v, space_mask));
        result = _mm256_or_si256(result, _mm256_and_si256(is_nl_v, nl_mask));

        __m256i zero_check = _mm256_cmpeq_epi8(result, _mm256_setzero_si256());
        __m256i other_mask = _mm256_set1_epi8(CLS_OTHER);
        result = _mm256_or_si256(result, _mm256_and_si256(zero_check, other_mask));

        _mm256_storeu_si256((__m256i*)(out_classes + i), result);
    }

    // Remaining bytes use scalar classify for UTF-8 awareness
    if (i < process_len) {
        for (uint32_t j = i; j < process_len; ++j) {
            uint8_t b = data[j];
            if (b >= 'A' && b <= 'Z') out_classes[j] = CLS_LETTER;
            else if (b >= 'a' && b <= 'z') out_classes[j] = CLS_LETTER;
            else if (b >= '0' && b <= '9') out_classes[j] = CLS_DIGIT;
            else if (b == ' ' || b == '\t') out_classes[j] = CLS_SPACE;
            else if (b == '\n' || b == '\r') out_classes[j] = CLS_NEWLINE;
            else if ((b & 0xE0) == 0xC0) {
                out_classes[j] = CLS_LETTER; // default multi-byte => letter
            } else {
                out_classes[j] = CLS_OTHER;
            }
        }
    }

    // Post-process: propagate UTF-8 starter class to continuation bytes
    propagate_utf8_classes(out_classes, data, process_len);

    out->count = process_len;
    return process_len;
#else
    return simd_classify_scalar(data, len, out);
#endif
}

uint32_t simd_boundaries_avx2(const uint8_t* data, const uint8_t* cls,
                              uint32_t len, WordBoundary* out, uint32_t max) {
    // Delegates to scalar — boundary detection involves variable-length
    // patterns that don't vectorize well.
    return simd_boundaries_scalar(data, cls, len, out, max);
}

bool simd_valid_utf8_avx2(const uint8_t* data, size_t len) {
#ifdef __AVX2__
    size_t i = 0;
    for (; i + 32 <= len; i += 32) {
        __m256i chunk = _mm256_loadu_si256((const __m256i*)(data + i));
        __m256i high = _mm256_set1_epi8((char)0x80);
        __m256i has_high = _mm256_and_si256(chunk, high);
        if (!_mm256_testz_si256(has_high, has_high)) break;
    }
    return simd_valid_utf8_scalar(data + i, len - i);
#else
    return simd_valid_utf8_scalar(data, len);
#endif
}
