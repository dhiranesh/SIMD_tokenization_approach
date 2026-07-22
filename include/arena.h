#ifndef TOKENIZER_ARENA_H
#define TOKENIZER_ARENA_H
/**
 * arena.h — Monotonic buffer arena allocator
 *
 * Why: Tokenization generates thousands of temporary string fragments per
 * second (pre-tokenized words, BPE symbols, merge results). std::string
 * allocations per fragment destroy cache locality and fragment the heap.
 *
 * Strategy: A monotonic arena allocates from large contiguous blocks and
 * only tracks a single bump-pointer. Resetting the arena costs O(1) —
 * no per-fragment free. Thread-local arenas eliminate contention.
 *
 * Key properties:
 *   - allocation: pointer bump (single load/add/store) — ~2ns
 *   - reset:      zero the bump pointer — ~1ns
 *   - growth:     exponential (2x) up to max_block, then cap
 *   - alignment:  naturally aligned to max_align_t (16 bytes)
 */
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <utility>
#include <vector>

class Arena {
public:
    explicit Arena(size_t block_size = 65536)
        : block_size_(block_size)
        , cur_block_(nullptr)
        , offset_(0)
        , end_(0)
        , total_allocated_(0) {}

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    Arena(Arena&& other) noexcept
        : block_size_(other.block_size_)
        , cur_block_(other.cur_block_)
        , offset_(other.offset_)
        , end_(other.end_)
        , blocks_(std::move(other.blocks_))
    {
        other.cur_block_ = nullptr;
        other.offset_ = 0;
        other.end_ = 0;
    }

    ~Arena() { reset(); }

    void reset() noexcept {
        for (size_t i = 1; i < blocks_.size(); ++i) {
            std::free(blocks_[i]);
        }
        if (!blocks_.empty()) {
            cur_block_ = blocks_[0];
            offset_ = 0;
            end_ = block_size_;
            blocks_.resize(1);
            block_sizes_.resize(1);
        } else {
            cur_block_ = nullptr;
            offset_ = 0;
            end_ = 0;
        }
        total_allocated_ = block_size_;
    }

    void* alloc(size_t bytes) noexcept {
        bytes = (bytes + 15) & ~size_t(15);
        if (offset_ + bytes <= end_) {
            void* ptr = cur_block_ + offset_;
            offset_ += bytes;
            return ptr;
        }
        return alloc_slow(bytes);
    }

    template <typename T>
    T* alloc_array(size_t count) noexcept {
        return static_cast<T*>(alloc(count * sizeof(T)));
    }

    char* dup_string(const char* data, size_t len) noexcept {
        char* buf = static_cast<char*>(alloc(len + 1));
        for (size_t i = 0; i < len; ++i) buf[i] = data[i];
        buf[len] = 0;
        return buf;
    }

    size_t used_bytes() const noexcept {
        size_t total = offset_;
        for (size_t i = 1; i < block_sizes_.size(); ++i) {
            total += block_sizes_[i];
        }
        return total;
    }

private:
    void* alloc_slow(size_t bytes) noexcept {
        size_t needed = (bytes > block_size_) ? bytes : block_size_;
        size_t next_pow2 = block_size_;
        // Double until we cover the request (cap at 64 MB)
        while (next_pow2 < needed && next_pow2 < (64u * 1024 * 1024)) {
            next_pow2 <<= 1;
        }
        if (next_pow2 < needed) next_pow2 = needed;
        void* block = std::malloc(next_pow2);
        if (!block) return nullptr;
        blocks_.push_back(static_cast<char*>(block));
        block_sizes_.push_back(next_pow2);
        cur_block_ = static_cast<char*>(block);
        offset_ = bytes;
        end_ = next_pow2;
        total_allocated_ += next_pow2;
        return block;
    }

    size_t block_size_;
    char* cur_block_;
    size_t offset_;
    size_t end_;
    std::vector<char*> blocks_;
    std::vector<size_t> block_sizes_;
    size_t total_allocated_ = 0;
};

struct ThreadLocalArena {
    static thread_local Arena instance;
};

#endif
