#ifndef TOKENIZER_BPE_ENGINE_H
#define TOKENIZER_BPE_ENGINE_H
/**
 * bpe_engine.h — Cache-optimized BPE merge engine
 *
 * Why: BPE merging is the hottest part of tokenization — each pre-tokenized
 * word goes through O(m log m) merge iterations. The naive implementation
 * creates a std::vector<std::string> per word, copies strings on every
 * merge, and does unordered_map lookups for pair ranks.
 *
 * Optimization strategies:
 *   1. Represent BPE symbols as integer IDs internally instead of strings.
 *      Compare and hash integers, not strings. String→ID resolution happens
 *      once at the start.
 *   2. Merge in-place using a compact linked-list representation:
 *      prev/next arrays of uint32_t indices, avoiding data copies.
 *   3. Cache common merge paths: a small set-associative cache mapping
 *      (start_id, end_id) pairs to the merged sequence of IDs.
 *   4. Arena allocation for temporary structures — no heap churn.
 */
#include <cstdint>
#include <cstddef>
#include <string_view>
#include <vector>
#include "arena.h"

class FlatVocab;

class BpeEngine {
public:
    explicit BpeEngine(const FlatVocab* vocab);

    // Encode a single pre-tokenized word (already byte-to-unicode mapped).
    // Returns view of token IDs in arena.
    std::string_view encode_word(std::string_view mapped_word, Arena& arena) const;

    // Pre-compute common merge results for cache warming
    void warmup_cache() const;

    // Clear the merge cache (e.g., between large batches)
    void clear_cache() const;

private:
    const FlatVocab* vocab_;

    // Direct-mapped hash table for (first_id, second_id) → rank
    // Flat array with linear probing. 0 key = empty slot.
    static constexpr uint32_t kRankTableSize = 262144;
    struct RankSlot { uint64_t key; uint32_t rank; };
    std::vector<RankSlot> merge_rank_table_;

    // Merge cache: (pair_key → merged_id)
    // Simple direct-mapped cache with 64K entries
    static constexpr uint32_t kCacheSize = 65536;
    mutable struct {
        uint64_t key = 0;
        uint32_t value = 0;
    } cache_[kCacheSize];

    uint32_t resolve_id(std::string_view token) const noexcept;
    uint32_t get_merge_rank(uint32_t first_id, uint32_t second_id) const noexcept;
    uint32_t get_cached_merge(uint64_t pair_key) const noexcept;
    void set_cached_merge(uint64_t pair_key, uint32_t result_id) const noexcept;
};

#endif
