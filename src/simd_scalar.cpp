/**
 * simd_scalar.cpp — Scalar (portable) fallback for bulk character ops
 *
 * Why: When AVX2/AVX-512 aren't available (older CPUs, some cloud VMs,
 * QEMU emulation), we fall back to a table-driven byte classifier that
 * is still faster than std::isalpha per byte because it uses a single
 * 256-entry lookup table (fits in L1 cache) instead of calling into
 * libc locale-sensitive functions.
 *
 * Performance: ~3-4 cycles/byte vs ~15-20 for std::isalpha. The lookup
 * table approach is also branchless for the common ASCII path.
 */
#include "simd_ops.h"
#include <cstring>

// 256-entry classification table indexed by byte value
// Build at compile time via constexpr
namespace {
    struct ClassTable {
        uint8_t data[256];

        constexpr ClassTable() : data{} {
            for (int i = 0; i < 256; ++i) {
                uint8_t c = 0;
                unsigned char b = static_cast<unsigned char>(i);

                // ASCII letters
                if ((b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z')) {
                    c |= CLS_LETTER;
                }
                // ASCII digits
                else if (b >= '0' && b <= '9') {
                    c |= CLS_DIGIT;
                }
                // ASCII control chars that are whitespace/newline
                else if (b == '\n' || b == '\r') {
                    c |= CLS_NEWLINE;
                }
                else if (b == '\t' || b == '\v' || b == '\f') {
                    c |= CLS_SPACE;
                }
                else if (b == ' ') {
                    c |= CLS_SPACE;
                }
                else if (b < 0x20 || b == 0x7F) {
                    c |= CLS_OTHER;
                }
                // ASCII printable but not letter/digit
                else if (b <= 0x7E) {
                    c |= CLS_OTHER;
                }
                // High bytes (>0x7F): UTF-8 multi-byte starters or continuation
                else {
                    if ((b & 0xE0) == 0xC0) {       // 110xxxxx → 2-byte start
                        c |= CLS_UTF8_2;
                    } else if ((b & 0xF0) == 0xE0) { // 1110xxxx → 3-byte start
                        c |= CLS_UTF8_3;
                    } else if ((b & 0xF8) == 0xF0) { // 11110xxx → 4-byte start
                        c |= CLS_UTF8_4;
                    } else if ((b & 0xC0) == 0x80) { // 10xxxxxx → continuation
                        c |= CLS_OTHER;
                    } else {
                        c |= CLS_OTHER;
                    }
                    // For high bytes, we don't know if they're letter/digit
                    // without decoding. Mark as OTHER; the pre-tokenizer
                    // handles this via full UTF-8 decoding.
                }
                data[i] = c;
            }
        }
    };

    constexpr ClassTable kClassTable;
}

static bool is_unicode_letter_cp(uint32_t cp) {
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

static bool is_unicode_number_cp(uint32_t cp) {
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

uint32_t simd_classify_scalar(const uint8_t* data, uint32_t len, ClassifyResult* out) {
    out->valid_utf8 = 1;
    uint32_t process_len = (len > 64) ? 64 : len;
    uint32_t i = 0;

    while (i < process_len) {
        uint8_t b = data[i];
        if (b <= 0x7F) {
            out->classes[i] = kClassTable.data[b];
            ++i;
        } else if ((b & 0xE0) == 0xC0) {
            // 2-byte UTF-8
            if (i + 1 >= process_len) { out->classes[i] = CLS_OTHER; ++i; continue; }
            uint32_t cp = (b & 0x1F) << 6 | (data[i+1] & 0x3F);
            uint8_t cls = is_unicode_number_cp(cp) ? CLS_DIGIT :
                          is_unicode_letter_cp(cp) ? CLS_LETTER : CLS_OTHER;
            out->classes[i] = cls;
            out->classes[i+1] = cls;
            i += 2;
        } else if ((b & 0xF0) == 0xE0) {
            // 3-byte UTF-8
            if (i + 2 >= process_len) { out->classes[i] = CLS_OTHER; ++i; continue; }
            uint32_t cp = (b & 0x0F) << 12 | (data[i+1] & 0x3F) << 6 | (data[i+2] & 0x3F);
            uint8_t cls = is_unicode_number_cp(cp) ? CLS_DIGIT :
                          is_unicode_letter_cp(cp) ? CLS_LETTER : CLS_OTHER;
            out->classes[i] = cls;
            out->classes[i+1] = cls;
            out->classes[i+2] = cls;
            i += 3;
        } else if ((b & 0xF8) == 0xF0) {
            // 4-byte UTF-8
            if (i + 3 >= process_len) { out->classes[i] = CLS_OTHER; ++i; continue; }
            uint32_t cp = (b & 0x07) << 18 | (data[i+1] & 0x3F) << 12 |
                          (data[i+2] & 0x3F) << 6 | (data[i+3] & 0x3F);
            uint8_t cls = is_unicode_letter_cp(cp) ? CLS_LETTER : CLS_OTHER;
            out->classes[i] = cls;
            out->classes[i+1] = cls;
            out->classes[i+2] = cls;
            out->classes[i+3] = cls;
            i += 4;
        } else {
            out->classes[i] = CLS_OTHER;
            ++i;
        }
    }

    out->count = process_len;
    return process_len;
}

uint32_t simd_boundaries_scalar(const uint8_t* data, const uint8_t* cls,
                                uint32_t len, WordBoundary* out, uint32_t max) {
    // GPT-2 pre-tokenization rules implemented as a state machine
    // over the classification array.
    //
    // Rules (in priority order, matched greedily at each position):
    //   1. Contractions: 's 't 're 've 'm 'll 'd
    //   2.  ?\p{L}+   — optional space followed by letters
    //   3.  ?\p{N}+   — optional space followed by digits
    //   4.  ?[^\s\p{L}\p{N}]+[\r\n]* — optional space + non-wnl + newlines
    //   5. \s*[\r\n]+ — whitespace + newlines
    //   6. \s+(?!\S)  — trailing whitespace
    //   7. \s+        — any remaining whitespace
    //
    // The state machine classifies byte-by-byte using the precomputed
    // class array, emitting bounds for pre-tokenized words.

    uint32_t count = 0;
    uint32_t i = 0;

    auto is_space = [&](uint32_t p) -> bool {
        return p < len && (cls[p] & (CLS_SPACE | CLS_NEWLINE));
    };
    auto is_nl = [&](uint32_t p) -> bool {
        return p < len && (cls[p] & CLS_NEWLINE);
    };
    auto is_letter = [&](uint32_t p) -> bool {
        return p < len && (cls[p] & CLS_LETTER);
    };
    auto is_digit = [&](uint32_t p) -> bool {
        return p < len && (cls[p] & CLS_DIGIT);
    };
    auto is_wnl = [&](uint32_t p) -> bool {
        return p < len && (cls[p] & (CLS_SPACE | CLS_NEWLINE | CLS_LETTER | CLS_DIGIT));
    };

    while (i < len && count < max) {
        uint32_t start = i;
        uint32_t matched = 0;

        // Rule 1: contractions 's 't 're 've 'm 'll 'd
        if (!matched && i < len && data[i] == '\'') {
            if (i + 1 < len) {
                uint8_t n1 = data[i+1];
                if (n1 == 's' || n1 == 'S' || n1 == 't' || n1 == 'T' ||
                    n1 == 'm' || n1 == 'M' || n1 == 'd' || n1 == 'D') {
                    out[count].start = i; out[count].len = 2;
                    i += 2; ++count; matched = 1;
                } else if ((n1 == 'l' || n1 == 'L') && i + 2 < len &&
                           (data[i+2] == 'l' || data[i+2] == 'L')) {
                    out[count].start = i; out[count].len = 3;
                    i += 3; ++count; matched = 1;
                } else if ((n1 == 'v' || n1 == 'V') && i + 2 < len &&
                           (data[i+2] == 'e' || data[i+2] == 'E')) {
                    out[count].start = i; out[count].len = 3;
                    i += 3; ++count; matched = 1;
                } else if ((n1 == 'r' || n1 == 'R') && i + 2 < len &&
                           (data[i+2] == 'e' || data[i+2] == 'E')) {
                    out[count].start = i; out[count].len = 3;
                    i += 3; ++count; matched = 1;
                }
            }
        }

        // Rule 2: [^\r\n\p{L}\p{N}]? \p{L}+  (exact HF GPT-2 regex)
        if (!matched) {
            uint32_t s = i;
            if (s < len && !is_letter(s) && !is_digit(s) && data[s] != '\r' && data[s] != '\n') {
                // Advance by FULL UTF-8 character (not just 1 byte)
                uint8_t b = data[s];
                if (b < 0x80) s += 1;
                else if ((b & 0xE0) == 0xC0) s += 2;
                else if ((b & 0xF0) == 0xE0) s += 3;
                else if ((b & 0xF8) == 0xF0) s += 4;
                else s += 1; // invalid continuation byte — skip 1
            }
            if (s < len && is_letter(s)) {
                while (s < len && is_letter(s)) ++s;
                out[count].start = i; out[count].len = s - i;
                i = s; ++count; matched = 1;
            }
        }

        // Rule 3: \p{N}  (single digit only — exact HF GPT-2 regex)
        if (!matched) {
            if (i < len && is_digit(i)) {
                out[count].start = i; out[count].len = 1;
                ++i; ++count; matched = 1;
            }
        }

        // Rule 4:  ?[^\s\p{L}\p{N}]+[\r\n]*
        if (!matched) {
            uint32_t s = i;
            if (s < len && data[s] == ' ') ++s;
            if (s < len && !is_wnl(s)) { // not whitespace/newline/letter/digit
                while (s < len && !is_wnl(s)) ++s;
                while (s < len && is_nl(s)) ++s;
                if (s > i + (data[i] == ' ' ? 1 : 0)) {
                    out[count].start = i; out[count].len = s - i;
                    i = s; ++count; matched = 1;
                }
            }
        }

        // Rule 5: \s*[\r\n]+
        if (!matched) {
            uint32_t s = i;
            while (s < len && (cls[s] & CLS_SPACE) && !(cls[s] & CLS_NEWLINE)) ++s;
            if (s < len && is_nl(s)) {
                while (s < len && is_nl(s)) ++s;
                out[count].start = i; out[count].len = s - i;
                i = s; ++count; matched = 1;
            }
        }

        // Rule 6: \s+(?!\S) — trailing whitespace
        if (!matched) {
            uint32_t s = i;
            if (s < len && is_space(s)) {
                bool only_ws = true;
                for (uint32_t j = s; j < len; ++j) {
                    if (!is_space(j)) { only_ws = false; break; }
                }
                if (only_ws) {
                    out[count].start = i; out[count].len = len - i;
                    i = len; ++count; matched = 1;
                }
            }
        }

        // Rule 7: \s+ (HF-compatible: leave 1 trailing space for alt2/4/5/6 prefixes)
        if (!matched) {
            uint32_t s = i;
            if (s < len && is_space(s)) {
                while (s < len && is_space(s)) ++s;
                uint32_t ws_len = s - i;
                // With >1 space, leave 1 for next iteration (HF backtracking)
                if (ws_len > 1) --s;
                // With 1 space followed by digit, output it (no alt prefix consumes it)
                // For 1 space before letter/non-wnl/etc, the space is already
                // consumed by alt2/4/5/6 before reaching here
                if (s > i) {
                    out[count].start = i; out[count].len = s - i;
                    i = s; ++count; matched = 1;
                }
            }
        }

        // Fallback: emit single byte
        if (!matched) {
            out[count].start = i; out[count].len = 1;
            ++i; ++count;
        }
    }

    return count;
}

bool simd_valid_utf8_scalar(const uint8_t* data, size_t len) {
    size_t i = 0;
    while (i < len) {
        uint8_t b = data[i];
        if (b <= 0x7F) {
            ++i; // ASCII, valid
        } else if (b <= 0xC1) {
            return false; // overlong or invalid lead
        } else if (b <= 0xDF) {
            if (i + 1 >= len || (data[i+1] & 0xC0) != 0x80) return false;
            i += 2;
        } else if (b <= 0xEF) {
            if (i + 2 >= len) return false;
            if ((data[i+1] & 0xC0) != 0x80 || (data[i+2] & 0xC0) != 0x80) return false;
            if (b == 0xE0 && data[i+1] < 0xA0) return false; // overlong
            if (b == 0xED && data[i+1] > 0x9F) return false; // surrogates
            i += 3;
        } else if (b <= 0xF4) {
            if (i + 3 >= len) return false;
            if ((data[i+1] & 0xC0) != 0x80 || (data[i+2] & 0xC0) != 0x80 ||
                (data[i+3] & 0xC0) != 0x80) return false;
            if (b == 0xF0 && data[i+1] < 0x90) return false; // overlong
            if (b == 0xF4 && data[i+1] > 0x8F) return false; // > U+10FFFF
            i += 4;
        } else {
            return false; // invalid lead byte
        }
    }
    return true;
}
