/**
 * tokenizer_v2.cpp — Fused Qwen3 BPE tokenizer implementation
 *
 * Pipeline (encode):
 *   1. Special token scan — locate and replace <|im_start|> etc. with IDs
 *   2. Pre-tokenization — SIMD-accelerated GPT-2 word boundary detection
 *   3. Byte-to-unicode mapping — each word's bytes → unicode chars (arena)
 *   4. BPE merge — integer-ID BPE with direct-mapped rank cache
 *   5. ID output — final token IDs
 *
 * Pipeline (decode):
 *   1. ID lookup — token string per ID
 *   2. Unicode-to-byte reverse mapping — recover raw bytes
 *   3. Concatenation — join decoded bytes into output string
 *
 * Memory: All per-batch temporaries use thread-local arenas. No per-token
 * heap allocations. Arena reset is O(1) between batches.
 */
#include "tokenizer_v2.h"
#include "simd_ops.h"
#include <cstring>
#include <algorithm>
#include <cstdio>

// Global SIMD dispatch — initialized at process startup
static bool g_simd_initialized = false;

QwenTokenizerV2::QwenTokenizerV2() {
    if (!g_simd_initialized) {
        init_simd_dispatch();
        g_simd_initialized = true;
    }
}

QwenTokenizerV2::~QwenTokenizerV2() {}

bool QwenTokenizerV2::load(const std::string& vocab_path, const std::string& merges_path) {
    if (!vocab_.load_vocab_json(vocab_path)) return false;
    if (!vocab_.load_merges_txt(merges_path)) return false;

    // Build the special token list from known Qwen tokens
    // (Qwen3 special tokens are at IDs 151643-151668)
    struct { const char* tok; int id; } builtins[] = {
        {"<|endoftext|>", 151643},
        {"<|im_start|>",  151644},
        {"<|im_end|>",    151645},
        {"<|object_ref_start|>", 151646},
        {"<|object_ref_end|>",   151647},
        {"<|box_start|>",  151648},
        {"<|box_end|>",    151649},
        {"<|quad_start|>", 151650},
        {"<|quad_end|>",   151651},
        {"<|vision_start|>", 151652},
        {"<|vision_end|>",   151653},
        {"<|vision_pad|>",   151654},
        {"<|image_pad|>",    151655},
        {"<|video_pad|>",    151656},
        {"<tool_call>",    151657},
        {"</tool_call>",   151658},
        {"<|fim_prefix|>", 151659},
        {"<|fim_middle|>", 151660},
        {"<|fim_suffix|>", 151661},
        {"<|fim_pad|>",    151662},
        {"<|repo_name|>",  151663},
        {"<|file_sep|>",   151664},
        {"<tool_response>",   151665},
        {"</tool_response>",  151666},
        {"<think>",  151667},
        {"</think>", 151668},
    };

    for (auto& b : builtins) {
        add_special_token(b.tok, b.id);
    }

    return true;
}

void QwenTokenizerV2::add_special_token(std::string_view token, int id) {
    special_tokens_.push_back({std::string(token), id});
    special_id_to_token_[id] = std::string(token);
    rebuild_special_list();
}

void QwenTokenizerV2::rebuild_special_list() {
    // Sort by length (longest first) for greedy matching
    std::sort(special_tokens_.begin(), special_tokens_.end(),
        [](const auto& a, const auto& b) {
            if (a.first.size() != b.first.size())
                return a.first.size() > b.first.size();
            return a.second < b.second;
        });
}

int QwenTokenizerV2::special_token_id(std::string_view token) const noexcept {
    for (const auto& [tok, id] : special_tokens_) {
        if (tok == token) return id;
    }
    return -1;
}

std::string_view QwenTokenizerV2::simd_backend() const noexcept {
    if (g_simd) return g_simd->name;
    return "not initialized";
}

uint32_t QwenTokenizerV2::encode_segment(
    std::string_view text, int* out_ids, uint32_t max_ids, Arena& arena) const
{
    // Allocate words array sized to accommodate all words in the segment.
    // Upper bound: every byte could start a new word in the worst case.
    uint32_t max_words = (uint32_t)text.size();
    if (max_words < 256) max_words = 256;
    PreTokenizedWord* words = arena.alloc_array<PreTokenizedWord>(max_words);
    uint32_t word_count = 0;
    pre_tokenizer_.pre_tokenize(text, arena, words, word_count, max_words);

    uint32_t id_count = 0;

    for (uint32_t w = 0; w < word_count && id_count < max_ids; ++w) {
        std::string_view mapped = pre_tokenizer_.map_bytes_to_unicode(
            words[w].original, arena);

        // Allocate initial symbol IDs from arena (upper bound = mapped byte count)
        uint32_t max_syms = (uint32_t)mapped.size();
        uint32_t* ids_buf = arena.alloc_array<uint32_t>(max_syms);
        uint32_t num_ids = 0;

        size_t pos = 0;
        while (pos < mapped.size() && num_ids < max_syms) {
            size_t start = pos;
            uint8_t b = (uint8_t)mapped[pos];
            size_t clen;
            if (b < 0x80) clen = 1;
            else if (b < 0xE0) clen = 2;
            else if (b < 0xF0) clen = 3;
            else clen = 4;
            pos += clen;

            std::string_view ch(mapped.data() + start, clen);
            int cid = vocab_.token_to_id(ch);
            if (cid >= 0) {
                ids_buf[num_ids] = (uint32_t)cid;
                ++num_ids;
            }
        }

        int merge_iter = 0;
        while (num_ids > 1) {
            if (++merge_iter > 10000) break;
            uint32_t best_rank = UINT32_MAX;
            int best_idx = -1;

            for (uint32_t i = 0; i < num_ids - 1; ++i) {
                int rr = vocab_.merge_rank_by_id(
                    (int)ids_buf[i], (int)ids_buf[i + 1]);
                if (rr >= 0 && (uint32_t)rr < best_rank) {
                    best_rank = (uint32_t)rr;
                    best_idx = (int)i;
                }
            }

            if (best_idx < 0 || best_rank == UINT32_MAX) break;

            // Look up merged token via string concatenation
            std::string_view first_str = vocab_.id_to_token((int)ids_buf[best_idx]);
            std::string_view second_str = vocab_.id_to_token((int)ids_buf[best_idx + 1]);
            uint32_t merged_len = (uint32_t)(first_str.size() + second_str.size());
            char* merge_buf = arena.alloc_array<char>(merged_len);
            std::memcpy(merge_buf, first_str.data(), first_str.size());
            std::memcpy(merge_buf + first_str.size(), second_str.data(), second_str.size());
            std::string_view merged_str(merge_buf, merged_len);

            int merged_id = vocab_.token_to_id(merged_str);
            if (merged_id < 0) break;

            ids_buf[best_idx] = (uint32_t)merged_id;
            for (uint32_t k = (uint32_t)best_idx + 1; k < num_ids - 1; ++k) {
                ids_buf[k] = ids_buf[k + 1];
            }
            --num_ids;
        }

        for (uint32_t i = 0; i < num_ids && id_count < max_ids; ++i) {
            out_ids[id_count++] = (int)ids_buf[i];
        }
    }

    return id_count;
}

std::vector<int> QwenTokenizerV2::encode(std::string_view text) const {
    std::vector<int> result;
    size_t reserve_size = text.size() / 2;
    if (reserve_size < 4096) reserve_size = 4096;
    result.reserve(reserve_size);

    Arena& arena = ThreadLocalArena::instance;
    arena.reset();

    struct Segment { std::string_view text; bool is_special; int special_id; };
    std::vector<Segment> segments;
    segments.reserve(64);

    size_t pos = 0;
    while (pos < text.size()) {
        bool found = false;
        for (const auto& [tok, id] : special_tokens_) {
            if (pos + tok.size() <= text.size() &&
                text.substr(pos, tok.size()) == tok) {
                segments.push_back({tok, true, id});
                pos += tok.size();
                found = true;
                break;
            }
        }
        if (!found) {
            size_t start = pos;
            while (pos < text.size()) {
                bool next_special = false;
                for (const auto& [tok, _] : special_tokens_) {
                    if (pos + tok.size() <= text.size() &&
                        text.substr(pos, tok.size()) == tok) {
                        next_special = true;
                        break;
                    }
                }
                if (next_special) break;
                ++pos;
            }
            segments.push_back({text.substr(start, pos - start), false, -1});
        }
    }

    for (size_t s = 0; s < segments.size(); ++s) {
        if (segments[s].is_special) {
            result.push_back(segments[s].special_id);
        } else {
            uint32_t max_temp = (uint32_t)segments[s].text.size();
            if (max_temp < 256) max_temp = 256;
            int* temp_ids = arena.alloc_array<int>(max_temp);
            uint32_t n = encode_segment(segments[s].text, temp_ids, max_temp, arena);
            for (uint32_t i = 0; i < n; ++i) result.push_back(temp_ids[i]);
        }
        arena.reset();
    }

    return result;
}

uint32_t QwenTokenizerV2::encode_into(
    std::string_view text, int* out_ids, uint32_t max_ids) const
{
    Arena& arena = ThreadLocalArena::instance;
    arena.reset();

    struct Segment { std::string_view text; bool is_special; int special_id; };
    std::vector<Segment> segments;
    segments.reserve(64);

    size_t pos = 0;
    while (pos < text.size()) {
        bool found = false;
        for (const auto& [tok, id] : special_tokens_) {
            if (pos + tok.size() <= text.size() &&
                text.substr(pos, tok.size()) == tok) {
                segments.push_back({tok, true, id});
                pos += tok.size();
                found = true;
                break;
            }
        }
        if (!found) {
            size_t start = pos;
            while (pos < text.size()) {
                bool next_special = false;
                for (const auto& [tok, _] : special_tokens_) {
                    if (pos + tok.size() <= text.size() &&
                        text.substr(pos, tok.size()) == tok) {
                        next_special = true;
                        break;
                    }
                }
                if (next_special) break;
                ++pos;
            }
            segments.push_back({text.substr(start, pos - start), false, -1});
        }
    }

    uint32_t id_count = 0;
    for (size_t s = 0; s < segments.size() && id_count < max_ids; ++s) {
        if (segments[s].is_special) {
            out_ids[id_count++] = segments[s].special_id;
        } else {
            uint32_t n = encode_segment(segments[s].text, out_ids + id_count,
                                        max_ids - id_count, arena);
            id_count += n;
        }
        arena.reset();
    }

    return id_count;
}

std::string QwenTokenizerV2::decode(const std::vector<int>& ids) const {
    return decode_span(ids.data(), (uint32_t)ids.size());
}

std::string QwenTokenizerV2::decode_span(const int* ids, uint32_t count) const {
    std::string result;
    result.reserve(count * 4);

    for (uint32_t i = 0; i < count; ++i) {
        std::string_view token = vocab_.id_to_token(ids[i]);
        if (token.empty()) {
            // Check if this is a special token not in FlatVocab
            auto it = special_id_to_token_.find(ids[i]);
            if (it != special_id_to_token_.end()) {
                result += it->second;
            }
            continue;
        }

        for (size_t j = 0; j < token.size(); ) {
            uint8_t b0 = (uint8_t)token[j];
            size_t char_len;
            if (b0 < 0x80) char_len = 1;
            else if (b0 < 0xE0) char_len = 2;
            else if (b0 < 0xF0) char_len = 3;
            else char_len = 4;

            int original_byte = pre_tokenizer_.reverse_lookup(
                token.data() + j, char_len);

            if (original_byte >= 0) {
                result += (char)original_byte;
            } else {
                result += token.substr(j, char_len);
            }

            j += char_len;
        }
    }

    return result;
}
