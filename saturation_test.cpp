#include "tokenizer_v2.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <string>
#include <algorithm>
#include <cstdint>

// Windows memory info
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

static double now_ms() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t.time_since_epoch()).count();
}

static size_t current_rss_kb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return pmc.WorkingSetSize / 1024;
#endif
    return 0;
}

struct SaturationConfig {
    const char* label;
    size_t text_size;
    int num_threads;
    int iterations;
};

struct ThreadResult {
    double encode_ms = 0;
    double decode_ms = 0;
    size_t tokens = 0;
    bool ok = false;
};

static void worker(const QwenTokenizerV2* tok, const std::string* text,
                   int iterations, ThreadResult* result) {
    result->ok = true;
    std::vector<int> last_ids;

    double t0 = now_ms();
    for (int i = 0; i < iterations; ++i) {
        auto ids = tok->encode(*text);
        if (ids.empty()) { result->ok = false; break; }
        result->tokens += ids.size();
        if (i > 0 && ids.size() != last_ids.size()) { result->ok = false; break; }
        last_ids = ids;
    }
    double t1 = now_ms();
    result->encode_ms = t1 - t0;

    t0 = now_ms();
    for (int i = 0; i < iterations; ++i) {
        auto decoded = tok->decode(last_ids);
        if (decoded != *text) { result->ok = false; break; }
    }
    t1 = now_ms();
    result->decode_ms = t1 - t0;
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL); // unbuffered

    if (argc < 2) {
        printf("usage: saturation_test <file1.md> [file2.md ...]\n");
        return 1;
    }

    // Load tokenizer
    QwenTokenizerV2 tokenizer;
    if (!tokenizer.load("D:/GPU inferance_tool/Qwen3_0.6B/vocab.json", "D:/GPU inferance_tool/Qwen3_0.6B/merges.txt")) {
        fprintf(stderr, "load failed\n");
        return 1;
    }

    // Load corpus files
    std::vector<std::string> corpus_chunks;
    size_t total_bytes = 0;
    for (int a = 1; a < argc; ++a) {
        FILE* f = std::fopen(argv[a], "rb");
        if (!f) continue;
        std::fseek(f, 0, SEEK_END);
        long sz = std::ftell(f);
        if (sz <= 0) { std::fclose(f); continue; }
        std::fseek(f, 0, SEEK_SET);
        std::string content((size_t)sz, '\0');
        if (std::fread(&content[0], 1, (size_t)sz, f) == (size_t)sz) {
            corpus_chunks.push_back(std::move(content));
            total_bytes += (size_t)sz;
        }
        std::fclose(f);
    }

    if (corpus_chunks.empty()) {
        printf("No corpus files loaded\n");
        return 1;
    }

    printf("Loaded %zu chunks, total %zu bytes\n", corpus_chunks.size(), total_bytes);

    // Configurations — sweep text sizes × thread counts
    SaturationConfig configs[] = {
        {"1K_1t",       1024,        1,  50},
        {"1K_2t",       1024,        2,  50},
        {"1K_4t",       1024,        4,  50},
        {"1K_8t",       1024,        8,  50},
        {"1K_14t",      1024,       14,  50},
        {"1K_20t",      1024,       20,  50},
        {"10K_1t",      10240,       1,  20},
        {"10K_2t",      10240,       2,  20},
        {"10K_4t",      10240,       4,  20},
        {"10K_8t",      10240,       8,  20},
        {"10K_14t",     10240,      14,  20},
        {"10K_20t",     10240,      20,  20},
        {"100K_1t",     102400,      1,  10},
        {"100K_2t",     102400,      2,  10},
        {"100K_4t",     102400,      4,  10},
        {"100K_8t",     102400,      8,  10},
        {"100K_14t",    102400,     14,  10},
        {"100K_20t",    102400,     20,  10},
        {"500K_1t",     512000,      1,  5},
        {"500K_2t",     512000,      2,  5},
        {"500K_4t",     512000,      4,  5},
        {"500K_8t",     512000,      8,  5},
        {"500K_14t",    512000,     14,  5},
        {"500K_20t",    512000,     20,  5},
        {"1M_1t",       1048576,     1,  3},
        {"1M_2t",       1048576,     2,  3},
        {"1M_4t",       1048576,     4,  3},
        {"1M_8t",       1048576,     8,  3},
        {"1M_14t",      1048576,    14,  3},
        {"1M_20t",      1048576,    20,  3},
        {"5M_1t",       5242880,     1,  2},
        {"5M_2t",       5242880,     2,  2},
        {"5M_4t",       5242880,     4,  2},
        {"5M_8t",       5242880,     8,  2},
        {"5M_14t",      5242880,    14,  2},
        {"5M_20t",      5242880,    20,  2},
        {"10M_1t",      10485760,    1,  1},
        {"10M_2t",      10485760,    2,  1},
        {"10M_4t",      10485760,    4,  1},
        {"10M_8t",      10485760,    8,  1},
        {"10M_14t",     10485760,   14,  1},
        {"10M_20t",     10485760,   20,  1},
        {"50M_1t",      52428800,    1,  1},
        {"50M_20t",     52428800,   20,  1},
    };

    size_t base_rss = current_rss_kb();
    printf("Base RSS: %zu KB\n\n", base_rss);

    printf("%-12s %8s %8s %10s %10s %10s %10s %8s %10s\n",
           "Config", "Size(KB)", "Threads", "Enc(MB/s)", "Dec(MB/s)",
           "Enc(ms)", "Tok(K)", "OK?", "RSS(KB)");

    for (auto& cfg : configs) {
        // Build test text of target size
        std::string text;
        text.reserve(cfg.text_size);
        while (text.size() < cfg.text_size) {
            for (auto& chunk : corpus_chunks) {
                if (text.size() >= cfg.text_size) break;
                size_t need = cfg.text_size - text.size();
                size_t take = (chunk.size() < need) ? chunk.size() : need;
                text.append(chunk, 0, take);
            }
        }
        size_t text_kb = text.size() / 1024;

        int n_iter = cfg.iterations;
        if (n_iter < 1) n_iter = 1;

        // Run concurrent test
        std::vector<std::thread> threads;
        std::vector<ThreadResult> results(cfg.num_threads);
        std::atomic<bool> any_failed{false};

        double t0 = now_ms();
        for (int ti = 0; ti < cfg.num_threads; ++ti) {
            threads.emplace_back([&, ti]() {
                worker(&tokenizer, &text, n_iter, &results[ti]);
                if (!results[ti].ok) any_failed = true;
            });
        }
        for (auto& th : threads) th.join();
        double elapsed = now_ms() - t0;

        // Compute aggregate
        double total_enc_ms = 0, total_dec_ms = 0;
        size_t total_tokens = 0;
        bool all_ok = true;
        for (auto& r : results) {
            total_enc_ms += r.encode_ms;
            total_dec_ms += r.decode_ms;
            total_tokens += r.tokens;
            if (!r.ok) all_ok = false;
        }
        if (any_failed) all_ok = false;

        double total_mb = (double)text.size() * n_iter * cfg.num_threads / (1024.0 * 1024.0);
        double enc_mbps = (elapsed > 0) ? (total_mb / (elapsed / 1000.0)) : 0;
        double dec_mbps = (elapsed > 0) ?
            ((double)text.size() * n_iter * cfg.num_threads / (1024.0 * 1024.0) / (elapsed / 1000.0)) : 0;

        size_t rss = current_rss_kb();
        std::string ok_str = all_ok ? (any_failed ? "PART" : "OK") : "FAIL";
        if (!all_ok && any_failed) ok_str = "FAIL";

        printf("%-12s %8zu %8d %10.1f %10.1f %10.1f %10zu %8s %10zu\n",
               cfg.label, text_kb, cfg.num_threads, enc_mbps, dec_mbps,
               elapsed, total_tokens / 1000, ok_str.c_str(), rss);
        fflush(stdout);

        // Check for OOM
        if (rss > 8 * 1024 * 1024) { // > 8GB
            printf("  WARNING: RSS > 8GB, may be near OOM\n");
        }
    }

    printf("\nSaturation test complete.\n");
    return 0;
}
