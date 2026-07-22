/**
 * pre_tokenizer.cpp — SIMD-accelerated GPT-2 regex pre-tokenizer
 *
 * Why separate from the main tokenizer: The pre-tokenizer is the first
 * stage in the pipeline and has different performance characteristics:
 * it's latency-sensitive (must process the entire input) but does simple
 * character-level work. Making it a separate stage allows the SIMD ops
 * to focus on bulk classification.
 *
 * The byte-to-unicode mapping tables are built here following the exact
 * GPT-2 algorithm. These tables define the reversible mapping between
 * raw bytes and the unicode characters used in BPE tokenization.
 */
#include "pre_tokenizer.h"
#include <cstring>
#include <algorithm>

PreTokenizer::PreTokenizer() {
    build_tables();
}

void PreTokenizer::build_tables() {
    // GPT-2 bytes_to_unicode() mapping:
    // Bytes 33-126 → Unicode 33-126 (printable ASCII)
    // Bytes 161-172 → Unicode 161-172 (¡-¬)
    // Bytes 174-255 → Unicode 174-255 (®-ÿ)
    // Remaining bytes (0-32, 127-160, 173) → Unicode 256+
    uint8_t bs[256];
    int bs_len = 0;

    // Phase 1: bytes that map to the same codepoint
    for (int b = 33; b <= 126; ++b) bs[bs_len++] = (uint8_t)b;
    for (int b = 161; b <= 172; ++b) bs[bs_len++] = (uint8_t)b;
    for (int b = 174; b <= 255; ++b) bs[bs_len++] = (uint8_t)b;

    int cs_first[256]; // unicode codepoints for phase 1 entries
    for (int i = 0; i < bs_len; ++i) cs_first[i] = bs[i];

    // Phase 2: remaining bytes mapped to 256+
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool found = false;
        for (int j = 0; j < bs_len; ++j) {
            if (bs[j] == (uint8_t)b) { found = true; break; }
        }
        if (!found) {
            bs[bs_len] = (uint8_t)b;
            cs_first[bs_len] = 256 + n;
            ++bs_len;
            ++n;
        }
    }

    // Build the mapping tables
    reverse_count_ = 0;

    for (int i = 0; i < 256; ++i) {
        byte_to_unicode_[i].clear();
    }

    for (int i = 0; i < bs_len; ++i) {
        uint32_t cp = (uint32_t)cs_first[i];

        // Encode unicode codepoint to UTF-8
        char utf8[5] = {0};
        int len;
        if (cp < 0x80) {
            utf8[0] = (char)cp; len = 1;
        } else if (cp < 0x800) {
            utf8[0] = (char)(0xC0 | (cp >> 6));
            utf8[1] = (char)(0x80 | (cp & 0x3F));
            len = 2;
        } else {
            utf8[0] = (char)(0xE0 | (cp >> 12));
            utf8[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
            utf8[2] = (char)(0x80 | (cp & 0x3F));
            len = 3;
        }

        byte_to_unicode_[bs[i]] = std::string(utf8, len);

        // Build reverse table entry
        ReverseEntry& re = reverse_table_[reverse_count_];
        std::memset(re.bytes, 0, 4);
        std::memcpy(re.bytes, utf8, len);
        re.byte_val = bs[i];
        ++reverse_count_;
    }
}

std::string_view PreTokenizer::pre_tokenize(
    std::string_view text, Arena& arena,
    PreTokenizedWord* out_words, uint32_t& out_count,
    uint32_t max_words) const
{
    out_count = 0;
    uint32_t len = (uint32_t)text.size();
    if (len == 0 || max_words == 0) return text;

    // Allocate classification array from arena for the FULL input
    uint8_t* cls_buf = arena.alloc_array<uint8_t>(len);

    // Classify input in small chunks (classify caps at 64 bytes per call)
    constexpr uint32_t kClassifyChunk = 64;
    uint32_t offset = 0;
    while (offset < len) {
        uint32_t chunk = (len - offset > kClassifyChunk) ? kClassifyChunk : (len - offset);
        ClassifyResult cr;
        g_simd->classify((const uint8_t*)(text.data() + offset), chunk, &cr);
        std::memcpy(cls_buf + offset, cr.classes, chunk);
        offset += chunk;
    }

    // Find word boundaries ONCE on the full classification array
    // (no cross-chunk state loss; full GPT-2 regex across entire input)
    WordBoundary* bounds = arena.alloc_array<WordBoundary>(max_words);
    uint32_t total_bounds = g_simd->find_boundaries(
        (const uint8_t*)text.data(), cls_buf, len, bounds, max_words);

    // Convert WordBoundary list to PreTokenizedWord
    out_count = total_bounds;
    for (uint32_t i = 0; i < total_bounds; ++i) {
        out_words[i].byte_start = bounds[i].start;
        out_words[i].byte_len   = bounds[i].len;
        out_words[i].original   = text.substr(bounds[i].start, bounds[i].len);
    }

    return text;
}

std::string_view PreTokenizer::map_bytes_to_unicode(
    std::string_view word, Arena& arena) const
{
    // Convert each byte of the word to its Unicode character via the
    // byte_to_unicode_ table. The output is a string of Unicode characters
    // (UTF-8 encoded) that the BPE engine will merge.

    // Estimate: worst case 4x expansion (each byte → 3-byte UTF-8 char)
    size_t max_out = word.size() * 3;
    char* buf = arena.alloc_array<char>(max_out);
    size_t out_pos = 0;

    for (size_t i = 0; i < word.size(); ++i) {
        uint8_t b = (uint8_t)word[i];
        const std::string& mapped = byte_to_unicode_[b];
        std::memcpy(buf + out_pos, mapped.data(), mapped.size());
        out_pos += mapped.size();
    }

    return std::string_view(buf, out_pos);
}

std::string_view PreTokenizer::unmap_unicode_to_bytes(
    std::string_view mapped, Arena& arena) const
{
    // Reverse mapping: each Unicode character → original byte
    size_t max_out = mapped.size(); // bytes are 1:1 or shorter
    char* buf = arena.alloc_array<char>(max_out);
    size_t out_pos = 0;

    for (size_t i = 0; i < mapped.size(); ) {
        uint8_t b0 = (uint8_t)mapped[i];

        size_t chlen;
        if (b0 < 0x80) chlen = 1;
        else if (b0 < 0xE0) chlen = 2;
        else if (b0 < 0xF0) chlen = 3;
        else chlen = 4;

        uint8_t byte_val = reverse_lookup(mapped.data() + i, chlen);

        if (byte_val != 0 || b0 == 0) {
            buf[out_pos++] = (char)byte_val;
            i += chlen;
        } else {
            buf[out_pos++] = mapped[i++];
        }
    }

    return std::string_view(buf, out_pos);
}

int PreTokenizer::reverse_lookup(const char* utf8_bytes, size_t len) const noexcept {
    uint32_t key = 0;
    std::memcpy(&key, utf8_bytes, len < 4 ? len : 4);

    for (uint32_t i = 0; i < reverse_count_; ++i) {
        uint32_t entry_key;
        std::memcpy(&entry_key, reverse_table_[i].bytes, 4);
        if (entry_key == key) {
            size_t entry_len = 0;
            for (int j = 0; j < 4; ++j) {
                if (reverse_table_[i].bytes[j] == 0) break;
                ++entry_len;
            }
            if (entry_len == len) return reverse_table_[i].byte_val;
        }
    }
    return -1;
}
