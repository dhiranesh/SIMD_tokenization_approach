#ifndef TOKENIZER_V2_H
#define TOKENIZER_V2_H
/**
 * tokenizer_v2.h — Fused Qwen3 BPE tokenizer with SIMD acceleration
 *
 * Why: Traditional tokenizers pipeline (pre-tokenize → byte-map → BPE →
 * lookup) with separate allocations per stage. This fused version:
 *   1. Processes input through SIMD-accelerated classification
 *   2. Extracts word boundaries without copying (string_view)
 *   3. Applies BPE merge with integer IDs instead of strings
 *   4. Outputs token IDs in a single pass
 *
 * Thread safety: encode() and decode() use thread-local arenas internally.
 * Multiple threads can call encode() concurrently without locking.
 * Vocabulary is read-only and shared.
 *
 * CPU dispatch: Best SIMD backend selected at process startup via CPUID.
 * Hot path calls through function pointers — zero micro-branches.
 */
#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <functional>

#include "arena.h"
#include "flat_vocab.h"
#include "bpe_engine.h"
#include "pre_tokenizer.h"

class QwenTokenizerV2 {
public:
    QwenTokenizerV2();
    ~QwenTokenizerV2();

    // Initialize: load vocabulary and merge files.
    // Must be called once before any encode/decode.
    // Returns false on failure (prints error to stderr).
    bool load(const std::string& vocab_path, const std::string& merges_path);

    // Encode: text → token IDs.
    // Thread-safe. Uses thread-local arena for temporaries.
    std::vector<int> encode(std::string_view text) const;

    // Encode into a pre-allocated buffer (zero-allocation path).
    // Returns number of IDs written.
    uint32_t encode_into(std::string_view text, int* out_ids, uint32_t max_ids) const;

    // Decode: token IDs → text.
    std::string decode(const std::vector<int>& ids) const;
    std::string decode_span(const int* ids, uint32_t count) const;

    // Register a special token (e.g., <|im_start|>, <|endoftext|>)
    void add_special_token(std::string_view token, int id);

    // Accessors
    int vocab_size() const noexcept { return (int)vocab_.vocab_size(); }
    int special_token_id(std::string_view token) const noexcept;

    // SIMD level info
    std::string_view simd_backend() const noexcept;

private:
    FlatVocab vocab_;
    PreTokenizer pre_tokenizer_;

    // Special tokens sorted by length (longest first) for greedy matching
    std::vector<std::pair<std::string, int>> special_tokens_;

    // Reverse map for decode: special token ID → token string
    std::unordered_map<int, std::string> special_id_to_token_;

    void rebuild_special_list();

    // Encode a single non-special segment
    uint32_t encode_segment(std::string_view text, int* out_ids, uint32_t max_ids,
                            Arena& arena) const;
};

#endif
