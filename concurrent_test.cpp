/**
 * concurrent_test.cpp — Multi-threaded stress test for tokenizer
 * Verifies thread safety, correctness, and performance under concurrent load.
 */
#include "tokenizer_v2.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <string>

static double now_ms() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t.time_since_epoch()).count();
}

struct ThreadResult {
    uint64_t tokens_encoded = 0;
    uint64_t tokens_decoded = 0;
    double encode_ms = 0;
    double decode_ms = 0;
    bool success = false;
    int first_id = -1;
    int last_id = -1;
};

static void worker(const QwenTokenizerV2* tok, const std::string* text,
                   int iterations, ThreadResult* result) {
    result->success = true;
    std::vector<int> last_ids;

    double t0 = now_ms();
    for (int i = 0; i < iterations; ++i) {
        auto ids = tok->encode(*text);
        if (ids.empty()) { result->success = false; break; }
        if (i == 0) {
            result->first_id = ids[0];
            result->last_id = ids.back();
        } else {
            // Verify deterministic: all iterations produce same IDs
            if ((int)ids.size() != (int)last_ids.size()) {
                result->success = false; break;
            }
            for (size_t j = 0; j < ids.size(); ++j) {
                if (ids[j] != last_ids[j]) { result->success = false; break; }
            }
            if (!result->success) break;
        }
        result->tokens_encoded += ids.size();
        last_ids = ids;
    }
    double t1 = now_ms();
    result->encode_ms = t1 - t0;

    // Decode
    t0 = now_ms();
    for (int i = 0; i < iterations; ++i) {
        auto decoded = tok->decode(last_ids);
        if (decoded != *text) {
            result->success = false;
            break;
        }
    }
    t1 = now_ms();
    result->decode_ms = t1 - t0;
    result->tokens_decoded = last_ids.size() * iterations;
}

int main() {
    std::printf("=== Concurrent Tokenizer Test ===\n\n");
    fflush(stdout);

    QwenTokenizerV2 tokenizer;
    if (!tokenizer.load("../Qwen3_0.6B/vocab.json", "../Qwen3_0.6B/merges.txt")) {
        std::fprintf(stderr, "Failed to load tokenizer!\n");
        return 1;
    }

    // Build test text (~100KB)
    std::string text;
    text.reserve(105000);
    for (int i = 0; i < 2000; ++i) {
        text += "The Qwen3 tokenizer is a Byte Pair Encoding tokenizer with byte-level fallback. "
                "It supports multiple languages including Chinese, Japanese, and Korean. "
                "Tokenization is the first step in natural language processing. "
                "42 + 100 = 142. Price: $9.99 (15% off). "
                "Hello 你好 Bonjour こんにちは!\n";
    }

    // Sequential baseline (single thread, 1 pass)
    std::printf("Sequential (single-thread) baseline:\n");
    double t0 = now_ms();
    auto ids_ref = tokenizer.encode(text);
    double t1 = now_ms();
    std::string decoded_ref = tokenizer.decode(ids_ref);
    bool roundtrip_ok = (decoded_ref == text);
    std::printf("  Encode: %.2f ms, %zu tokens\n", t1 - t0, ids_ref.size());
    std::printf("  Roundtrip: %s\n\n", roundtrip_ok ? "OK" : "FAIL");

    // Concurrent test
    const int thread_counts[] = {2, 4, 8, 16};
    const int iterations_per_thread = 50;

    for (int num_threads : thread_counts) {
        std::printf("--- %d threads (%d iter/thread) ---\n", num_threads, iterations_per_thread);

        std::vector<std::thread> threads;
        std::vector<ThreadResult> results(num_threads);
        std::atomic<bool> any_failed{false};
        std::atomic<uint64_t> total_encoded{0};
        std::atomic<uint64_t> total_decoded{0};

        t0 = now_ms();
        for (int ti = 0; ti < num_threads; ++ti) {
            threads.emplace_back([&, ti]() {
                worker(&tokenizer, &text, iterations_per_thread, &results[ti]);
                if (!results[ti].success) any_failed = true;
                total_encoded += results[ti].tokens_encoded;
                total_decoded += results[ti].tokens_decoded;
            });
        }
        for (auto& th : threads) th.join();
        double elapsed = now_ms() - t0;

        // Verify all threads produced same first/last IDs
        int ref_first = results[0].first_id;
        int ref_last = results[0].last_id;
        bool ids_consistent = true;
        double total_encode_ms = 0, total_decode_ms = 0;
        for (int ti = 0; ti < num_threads; ++ti) {
            if (results[ti].first_id != ref_first || results[ti].last_id != ref_last)
                ids_consistent = false;
            total_encode_ms += results[ti].encode_ms;
            total_decode_ms += results[ti].decode_ms;
            std::printf("  Thread %2d: encode=%6.0f ms decode=%6.0f ms tokens=%llu %s\n",
                       ti, results[ti].encode_ms, results[ti].decode_ms,
                       (unsigned long long)results[ti].tokens_encoded,
                       results[ti].success ? "OK" : "FAIL");
        }

        double text_kb = text.size() / 1024.0;
        double total_mb = text.size() * iterations_per_thread * num_threads / (1024.0 * 1024.0);
        double enc_mbps = total_mb / (elapsed / 1000.0);

        std::printf("  Summary: %d threads, %.2f sec wall, %.1f MB total, %.1f MB/s wall\n",
                   num_threads, elapsed / 1000.0, total_mb, enc_mbps);
        std::printf("  IDs consistent across threads: %s\n", ids_consistent ? "YES" : "NO");
        std::printf("  Any failures: %s\n", any_failed ? "YES" : "NO");
        std::printf("\n");
        fflush(stdout);
    }

    // Verify roundtrip
    std::printf("=== Verification ===\n");
    // Quick single-thread correctness check on a small text
    std::string small = "The quick brown fox jumps over the lazy dog. 42 + 100 = 142.";
    auto small_ids = tokenizer.encode(small);
    std::string small_decoded = tokenizer.decode(small_ids);
    bool small_ok = (small_decoded == small);
    std::printf("Small roundtrip: %s (%zu tokens)\n", small_ok ? "OK" : "FAIL", small_ids.size());

    std::printf("\nConcurrent test complete.\n");
    return 0;
}
