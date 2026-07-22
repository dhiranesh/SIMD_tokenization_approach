#include "include/tokenizer_v2.h"
#include <cstdio>
#include <chrono>
#include <string>
#include <vector>

static double now_ms() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t.time_since_epoch()).count();
}

struct TestCase {
    const char* name;
    const char* text;
};

int main() {
    std::printf("=== Qwen3 Tokenizer V2 (SIMD Accelerated) ===\n\n");
    fflush(stdout);

    QwenTokenizerV2 tokenizer;
    bool ok = tokenizer.load(
        "../Qwen3_0.6B/vocab.json",
        "../Qwen3_0.6B/merges.txt");

    if (!ok) {
        std::fprintf(stderr, "Failed to load tokenizer!\n");
        return 1;
    }

    std::printf("SIMD backend: %s\n", std::string(tokenizer.simd_backend()).c_str());
    std::printf("Vocab size: %d\n\n", tokenizer.vocab_size());
    fflush(stdout);

    TestCase tests[] = {
        {"Basic English",   "Hello, world! This is a test."},
        {"Chinese",         "你好，世界！这是一个测试。"},
        {"Numbers & symbols","42 + 100 = 142. Price: $9.99 (15% off)"},
        {"Multi-lingual",   "Hello 你好 Bonjour こんにちは 안녕하세요"},
        {"Contractions",    "I don't like it. She's happy. We'll see. They've done it."},
        {"Whitespace",      "line1\n\nline2\n  indented\n\t tabbed"},
        {"Special tokens",  "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n"},
        {"Mixed",           "Test: 123 + 456 = 579. <|endoftext|> Final."},
    };

    int total_match = 0;
    for (auto& tc : tests) {
        std::vector<int> ids = tokenizer.encode(tc.text);
        std::string decoded = tokenizer.decode(ids);
        bool match = (decoded == tc.text);

        std::printf("  [%s] %s\n", match ? "PASS" : "FAIL", tc.name);
        if (ids.size() <= 25) {
            std::printf("    IDs: ");
            for (int id : ids) std::printf("%d ", id);
            std::printf("\n");
        } else {
            std::printf("    IDs: %zu tokens\n", ids.size());
        }
        if (!match) {
            std::printf("    Decoded: \"%s\"\n", decoded.c_str());
        }
        fflush(stdout);
        if (match) ++total_match;
    }
    std::printf("\n%d/%zu tests passed\n\n", total_match,
                (int)(sizeof(tests)/sizeof(tests[0])));
    fflush(stdout);

    // Benchmark
    std::printf("=== Benchmark ===\n");
    fflush(stdout);

    std::string bench_text;
    bench_text.reserve(450000);
    for (int i = 0; i < 10000; ++i) {
        bench_text += "The quick brown fox jumps over the lazy dog. ";
    }
    std::printf("Input: %zu bytes\n", bench_text.size());
    fflush(stdout);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        std::printf("w%d ", i); fflush(stdout);
        auto ids = tokenizer.encode(bench_text);
        std::printf("(%zu) ", ids.size()); fflush(stdout);
    }
    std::printf("\nWarmup done.\n");
    fflush(stdout);

    const int iterations = 20;
    double t0 = now_ms();
    uint64_t total_tokens = 0;

    for (int i = 0; i < iterations; ++i) {
        auto ids = tokenizer.encode(bench_text);
        total_tokens += ids.size();
    }

    double t1 = now_ms();
    double elapsed = t1 - t0;
    double bytes_per_sec = (bench_text.size() * iterations) / (elapsed / 1000.0);
    double tokens_per_sec = total_tokens / (elapsed / 1000.0);

    std::printf("Iterations: %d\n", iterations);
    std::printf("Total tokens: %llu\n", (unsigned long long)total_tokens);
    std::printf("Time: %.2f ms (avg %.3f ms/iter)\n", elapsed, elapsed / iterations);
    std::printf("Throughput: %.1f MB/s\n", bytes_per_sec / (1024.0 * 1024.0));
    std::printf("Throughput: %.0f tokens/s\n", tokens_per_sec);
    fflush(stdout);

    // Decode benchmark
    auto ids = tokenizer.encode(bench_text);
    t0 = now_ms();
    for (int i = 0; i < iterations; ++i) {
        tokenizer.decode(ids);
    }
    t1 = now_ms();
    elapsed = t1 - t0;
    std::printf("\nDecode: %.2f ms (avg %.3f ms)\n", elapsed, elapsed / iterations);
    fflush(stdout);

    return 0;
}
