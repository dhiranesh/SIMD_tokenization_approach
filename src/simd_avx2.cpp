/**
 * simd_avx2.cpp — AVX2-accelerated character classification
 *
 * Compiled separately with -mavx2 (GCC/Clang) or /arch:AVX2 (MSVC).
 * Falls back to scalar if AVX2 not available at compile time.
 */
#include "simd_ops.h"
#include "unicode_props.h"
#include <cstring>

#ifdef __AVX2__
#include <immintrin.h>
#endif

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
            uint8_t cls = is_unicode_number_cp(cp) ? CLS_DIGIT :
                          is_unicode_letter_cp(cp) ? CLS_LETTER : CLS_OTHER;
            out_classes[j] = cls;
            out_classes[j+1] = cls;
            j += 2;
        } else if ((b & 0xF0) == 0xE0) {
            if (j + 2 >= process_len) { out_classes[j] = CLS_OTHER; ++j; continue; }
            uint32_t cp = (b & 0x0F) << 12 | (data[j+1] & 0x3F) << 6 | (data[j+2] & 0x3F);
            uint8_t cls = is_unicode_number_cp(cp) ? CLS_DIGIT :
                          is_unicode_letter_cp(cp) ? CLS_LETTER : CLS_OTHER;
            for (uint32_t k = 0; k < 3 && j + k < process_len; ++k)
                out_classes[j + k] = cls;
            j += 3;
        } else if ((b & 0xF8) == 0xF0) {
            if (j + 3 >= process_len) { out_classes[j] = CLS_OTHER; ++j; continue; }
            uint32_t cp = (b & 0x07) << 18 | (data[j+1] & 0x3F) << 12 |
                          (data[j+2] & 0x3F) << 6 | (data[j+3] & 0x3F);
            uint8_t cls = is_unicode_letter_cp(cp) ? CLS_LETTER : CLS_OTHER;
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
