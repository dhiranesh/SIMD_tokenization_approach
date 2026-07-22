/**
 * flat_vocab.cpp — Flat vocabulary construction and lookup
 *
 * Why: The vocabulary has 151K+ entries. A hash map with individually
 * allocated strings causes ~3-4 cache misses per lookup. A flat sorted
 * array of (hash, id, offset, len) tuples fits in L2 cache (~2.4MB for
 * 151K * 16 bytes) and binary search adds ~18 iterations (one cache miss
 * per ~4 iterations). This is ~4x faster than unordered_map for this scale.
 *
 * JSON parsing: We use a minimal state-machine parser (no dependency on
 * nlohmann/json or similar). It handles the specific format of HuggingFace
 * vocab.json and merges.txt files.
 */
#include "flat_vocab.h"
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <sstream>

uint32_t FlatVocab::fnv1a_hash(std::string_view s) noexcept {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < s.size(); ++i) {
        h ^= static_cast<uint8_t>(s[i]);
        h *= 16777619u;
    }
    return h;
}

bool FlatVocab::load_vocab_json(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::fprintf(stderr, "[FlatVocab] Failed to open: %s\n", path.c_str());
        return false;
    }

    size_t filesize = (size_t)file.tellg();
    file.seekg(0);
    std::string content(filesize, '\0');
    file.read(content.data(), filesize);
    file.close();

    // Temporary storage during parsing
    struct RawEntry {
        std::string token;
        int id;
    };
    std::vector<RawEntry> raw_entries;
    raw_entries.reserve(160000); // slightly over capacity

    // Minimal JSON parser for {"string": int, "string": int, ...}
    enum State { KEY, COLON, VALUE, NEXT };
    State state = KEY;
    std::string current_key;
    bool in_string = false;
    bool escape = false;

    for (size_t i = 0; i < content.size(); ++i) {
        char c = content[i];

        if (escape) {
            current_key += c;
            escape = false;
            continue;
        }

        if (c == '\\' && in_string) {
            escape = true;
            continue;
        }

        if (c == '"') {
            if (!in_string) {
                in_string = true;
                current_key.clear();
            } else {
                in_string = false;
                state = COLON;
            }
            continue;
        }

        if (in_string) {
            current_key += c;
            continue;
        }

        if (c == ':') {
            state = VALUE;
            continue;
        }

        if (state == VALUE && (c == '-' || (c >= '0' && c <= '9'))) {
            // Parse integer value
            size_t start = i;
            while (i + 1 < content.size() && content[i + 1] >= '0' && content[i + 1] <= '9') ++i;
            int value = std::stoi(content.substr(start, i - start + 1));
            raw_entries.push_back({std::move(current_key), value});
            current_key.clear();
            state = NEXT;
            continue;
        }

        if (c == ',' || c == '}') {
            state = KEY;
        }
    }

    if (raw_entries.empty()) {
        std::fprintf(stderr, "[FlatVocab] No entries found in vocab\n");
        return false;
    }

    // Sort raw entries by ID to build id_to_token
    std::sort(raw_entries.begin(), raw_entries.end(),
        [](const RawEntry& a, const RawEntry& b) { return a.id < b.id; });

    // Build id_offsets and string_data for id_to_token
    token_count_ = raw_entries.size();
    id_offsets_.resize(token_count_ + 1);

    // Calculate total string data size
    size_t total_str_len = 0;
    for (size_t i = 0; i < raw_entries.size(); ++i) {
        total_str_len += raw_entries[i].token.size() + 1; // +1 for null
    }
    string_data_.resize(total_str_len);

    size_t str_offset = 0;
    for (size_t i = 0; i < raw_entries.size(); ++i) {
        id_offsets_[i] = (uint32_t)str_offset;
        std::memcpy(string_data_.data() + str_offset, raw_entries[i].token.data(),
                    raw_entries[i].token.size());
        str_offset += raw_entries[i].token.size();
        string_data_[str_offset] = 0;
        str_offset += 1;
    }
    id_offsets_[token_count_] = (uint32_t)str_offset;

    // Build token_to_id entries sorted by hash
    entries_.reserve(token_count_);
    for (size_t i = 0; i < raw_entries.size(); ++i) {
        std::string_view tok = id_to_token((int)i);
        TokenEntry te;
        te.hash   = fnv1a_hash(tok);
        te.id     = (uint32_t)i;
        te.offset = id_offsets_[i];
        te.len    = (uint32_t)tok.size();
        entries_.push_back(te);
    }

    // Sort by hash for binary search
    std::sort(entries_.begin(), entries_.end(),
        [](const TokenEntry& a, const TokenEntry& b) { return a.hash < b.hash; });

    std::fprintf(stderr, "[FlatVocab] Loaded %zu tokens (%zu bytes string data)\n",
                 token_count_, total_str_len);
    return true;
}

bool FlatVocab::load_merges_txt(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::fprintf(stderr, "[FlatVocab] Failed to open merges: %s\n", path.c_str());
        return false;
    }

    std::fprintf(stderr, "[FlatVocab] Reading merges...\n");

    // We need to store merge component strings for later ID resolution.
    // Append them to string_data_ and store offsets.
    std::string line;
    bool first_line = true;
    uint32_t rank = 0;

    while (std::getline(file, line)) {
        if (first_line) {
            first_line = false;
            if (line.find("#version:") == 0 || line.find("#") == 0) continue;
        }
        if (line.empty()) continue;

        size_t space_pos = line.find(' ');
        if (space_pos == std::string::npos) continue;

        MergePair mp;
        mp.rank = rank++;

        // Store first component string
        mp.first_str_len = (uint32_t)space_pos;
        mp.first_str_off = (uint32_t)string_data_.size();
        string_data_.insert(string_data_.end(), line.begin(), line.begin() + space_pos);
        string_data_.push_back(0); // null terminator

        // Store second component string
        mp.second_str_len = (uint32_t)(line.size() - space_pos - 1);
        mp.second_str_off = (uint32_t)string_data_.size();
        string_data_.insert(string_data_.end(), line.begin() + space_pos + 1, line.end());
        string_data_.push_back(0);

        // Compute hashes
        std::string_view first_str(string_data_.data() + mp.first_str_off, mp.first_str_len);
        std::string_view second_str(string_data_.data() + mp.second_str_off, mp.second_str_len);
        mp.first_hash  = fnv1a_hash(first_str);
        mp.second_hash = fnv1a_hash(second_str);

        // Resolve component IDs from vocab (may fail for some if not yet loaded)
        mp.first_id  = (uint32_t)token_to_id(first_str);
        mp.second_id = (uint32_t)token_to_id(second_str);

        // Resolve merged token ID
        std::string merged_str(first_str);
        merged_str += second_str;
        int merged_id = token_to_id(merged_str);
        mp.merged_id = (merged_id >= 0) ? (uint32_t)merged_id : UINT32_MAX;

        merge_pairs_.push_back(mp);
    }

    std::fprintf(stderr, "[FlatVocab] Parsed %zu merge pairs, building hash table...\n", merge_pairs_.size());
    // Build hash table for O(1) merge rank lookup
    // Use the golden-ratio multiplicative hash to properly mix both 32-bit
    // components into the slot index (avoiding the 'key % power-of-2' trap
    // that only uses low bits of second_id).
    rank_hash_.resize(kRankHashSize, RankSlot{0, 0});
    for (size_t i = 0; i < merge_pairs_.size(); ++i) {
        uint64_t key = ((uint64_t)merge_pairs_[i].first_id << 32) | merge_pairs_[i].second_id;
        if (key == 0) continue;
        uint64_t mix = key * 0x9E3779B97F4A7C15ULL;
        uint32_t slot = (uint32_t)(mix >> (64 - 19));
        while (rank_hash_[slot].key != 0 && rank_hash_[slot].key != key) {
            slot = (slot + 1) & (kRankHashSize - 1);
        }
        if (rank_hash_[slot].key == 0) {
            rank_hash_[slot].key = key;
            rank_hash_[slot].rank = merge_pairs_[i].rank;
        }
    }

    std::fprintf(stderr, "[FlatVocab] Loaded %zu merge rules\n", merge_pairs_.size());
    return true;
}

int FlatVocab::token_to_id(std::string_view token) const noexcept {
    if (entries_.empty()) return -1;
    uint32_t h = fnv1a_hash(token);

    // Binary search on hash
    auto it = std::lower_bound(entries_.begin(), entries_.end(), h,
        [](const TokenEntry& e, uint32_t hash) { return e.hash < hash; });

    // Linear scan for matching hash (handles collisions)
    while (it != entries_.end() && it->hash == h) {
        std::string_view candidate(string_data_.data() + it->offset, it->len);
        if (candidate == token) return (int)it->id;
        ++it;
    }

    return -1;
}

std::string_view FlatVocab::id_to_token(int id) const noexcept {
    if (id < 0 || (size_t)id >= token_count_) return {};
    uint32_t off = id_offsets_[id];
    uint32_t len = id_offsets_[id + 1] - off - 1; // -1 for null terminator
    return std::string_view(string_data_.data() + off, len);
}

int FlatVocab::merge_rank(std::string_view first, std::string_view second) const noexcept {
    uint32_t h1 = fnv1a_hash(first);
    uint32_t h2 = fnv1a_hash(second);

    // Optimization: use merge_pairs_ index as a direct-mapped cache.
    // Since merge_pairs_ is sorted by rank (which is the load order from merges.txt),
    // we can build a hash table for O(1) lookup.
    // For now, use a fast linear scan limited by first_hash matching only.
    // This is still slow for the BPE hot loop. In production, use the
    // BpeEngine's rank table instead.
    //
    // Fast path: only scan entries matching first_hash
    for (size_t i = 0; i < merge_pairs_.size(); ++i) {
        if (merge_pairs_[i].first_hash == h1) {
            if (merge_pairs_[i].second_hash == h2) {
                return (int)merge_pairs_[i].rank;
            }
        }
    }
    return -1;
}

int FlatVocab::merge_rank_by_id(int first_id, int second_id) const noexcept {
    uint64_t key = ((uint64_t)(uint32_t)first_id << 32) | (uint32_t)second_id;
    if (key == 0) return -1;
    uint64_t mix = key * 0x9E3779B97F4A7C15ULL;
    uint32_t slot = (uint32_t)(mix >> (64 - 19));
    while (rank_hash_[slot].key != 0) {
        if (rank_hash_[slot].key == key) return (int)rank_hash_[slot].rank;
        slot = (slot + 1) & (kRankHashSize - 1);
    }
    return -1;
}

void FlatVocab::add_special_token(std::string_view token, int id) {
    // Check if already exists
    if (token_to_id(token) >= 0) return;

    // We don't dynamically resize the flat arrays for special tokens.
    // Instead, special tokens are handled separately in the tokenizer.
    // This is here for completeness.
}
