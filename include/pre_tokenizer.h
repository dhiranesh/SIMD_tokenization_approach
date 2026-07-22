#ifndef TOKENIZER_PRE_TOKENIZER_H
#define TOKENIZER_PRE_TOKENIZER_H
/**
 * pre_tokenizer.h — SIMD-accelerated GPT-2 pre-tokenization
 *
 * Why: GPT-2 regex pre-tokenization splits text into words by matching
 * a complex regex pattern. Doing this byte-by-byte with std::regex is
 * slow (~100ns/byte). Using SIMD bulk classification + a state machine
 * reduces this to ~3-5ns/byte.
 *
 * What it produces: A list of WordBoundary structs pointing into the
 * original input buffer (no copies). The pre-tokenizer does NOT allocate
 * or copy string data — it returns views into the source.
 *
 * The byte-to-unicode mapping is applied on each word before BPE:
 *   input byte → bytes_to_unicode[byte] → unicode character string
 *
 * This is done lazily in the encode loop, not in the pre-tokenizer.
 */
#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include "simd_ops.h"
#include "arena.h"

struct PreTokenizedWord {
    std::string_view original;   // view into input buffer
    uint32_t byte_start;         // offset in original input
    uint32_t byte_len;
};

class PreTokenizer {
public:
    PreTokenizer();

    // Split text into pre-tokenized words using GPT-2 regex rules.
    // Returns word boundaries in the provided arena. The words themselves
    // are views into the original input buffer (zero copy).
    std::string_view pre_tokenize(std::string_view text, Arena& arena,
                                  PreTokenizedWord* out_words, uint32_t& out_count,
                                  uint32_t max_words) const;

    // Apply byte-to-unicode mapping to a single word.
    // Returns the mapped string allocated in the arena.
    std::string_view map_bytes_to_unicode(std::string_view word, Arena& arena) const;

    // Decode: convert unicode-mapped token back to raw bytes
    std::string_view unmap_unicode_to_bytes(std::string_view mapped, Arena& arena) const;

private:
    // The byte-to-unicode mapping tables (GPT-2 style)
    // byte_to_unicode[b] → unicode character string (UTF-8 encoded)
    std::string byte_to_unicode_[256];

    // Reverse mapping for decode: maps UTF-8 character bytes → original byte
    // Stored as parallel arrays for binary search by UTF-8 bytes
    struct ReverseEntry {
        uint8_t bytes[4]; // UTF-8 encoding (zero-padded)
        uint8_t byte_val; // original byte value
    };
    ReverseEntry reverse_table_[256];
    uint32_t reverse_count_;

    void build_tables();

public:
    // Reverse mapping for decode. Returns -1 if not found.
    int reverse_lookup(const char* utf8_bytes, size_t len) const noexcept;
};

#endif
