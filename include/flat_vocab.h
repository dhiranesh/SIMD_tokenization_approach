#ifndef TOKENIZER_FLAT_VOCAB_H
#define TOKENIZER_FLAT_VOCAB_H
/**
 * flat_vocab.h — Flat, cache-friendly vocabulary storage
 *
 * Why: std::unordered_map<std::string, int> is pointer-heavy (each entry
 * is a separate allocation for the string data + hash table node) and
 * causes cache misses on every lookup. A flat array of sorted entries
 * with inline string data is ~5-10x more cache-friendly.
 *
 * Layout in memory (token_to_id):
 *   [header][entry_0][entry_1]...[entry_N]
 *   Each entry = 16 bytes:
 *     uint32_t hash   — 32-bit FNV-1a hash
 *     uint32_t id     — token ID
 *     uint32_t offset — byte offset of inline token string from buffer start
 *     uint32_t len    — byte length of token string
 */
#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

class FlatVocab {
public:
    struct MergePair {
        uint32_t first_hash;
        uint32_t second_hash;
        uint32_t first_id;
        uint32_t second_id;
        uint32_t merged_id;
        uint32_t rank;
        uint32_t first_str_off;
        uint32_t second_str_off;
        uint32_t first_str_len;
        uint32_t second_str_len;
    };

    bool load_vocab_json(const std::string& path);
    bool load_merges_txt(const std::string& path);

    int token_to_id(std::string_view token) const noexcept;
    std::string_view id_to_token(int id) const noexcept;

    int merge_rank(std::string_view first, std::string_view second) const noexcept;
    int merge_rank_by_id(int first_id, int second_id) const noexcept;

    size_t vocab_size() const noexcept { return token_count_; }
    size_t merge_count() const noexcept { return merge_pairs_.size(); }

    const MergePair* merge_pairs() const noexcept { return merge_pairs_.data(); }
    size_t merge_pair_count() const noexcept { return merge_pairs_.size(); }

    void add_special_token(std::string_view token, int id);
    bool is_initialized() const noexcept { return token_count_ > 0; }

    const void* data() const noexcept { return buffer_.data(); }
    size_t data_size() const noexcept { return buffer_.size(); }

private:
    struct TokenEntry {
        uint32_t hash;
        uint32_t id;
        uint32_t offset;
        uint32_t len;
    };

    // Direct-mapped hash table for O(1) merge rank lookup
    // Key: (first_id << 32) | second_id, Value: rank
    static constexpr uint32_t kRankHashSize = 524288; // 2x merge count, power of 2
    struct RankSlot { uint64_t key; uint32_t rank; };
    std::vector<RankSlot> rank_hash_;

    std::vector<char> buffer_;
    std::vector<TokenEntry> entries_;
    std::vector<uint32_t> id_offsets_;
    std::vector<char> string_data_;
    std::vector<MergePair> merge_pairs_;

    size_t token_count_ = 0;

    static uint32_t fnv1a_hash(std::string_view s) noexcept;
};

#endif
