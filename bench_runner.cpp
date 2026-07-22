#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include "tokenizer_v2.h"

struct BenchInput {
    std::string text;
    int warmup;
    int iterations;
};

static BenchInput read_input(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    uint32_t len;
    if (std::fread(&len, 4, 1, f) != 1) { std::perror("read len"); std::exit(1); }
    std::string text(len, '\0');
    if (len > 0 && std::fread(&text[0], 1, len, f) != len) { std::perror("read text"); std::exit(1); }
    uint32_t warmup, iters;
    if (std::fread(&warmup, 4, 1, f) != 1) warmup = 3;
    if (std::fread(&iters, 4, 1, f) != 1) iters = 10;
    std::fclose(f);
    return {std::move(text), (int)warmup, (int)iters};
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: bench_runner <input.bin>\n");
        return 1;
    }
    BenchInput bi = read_input(argv[1]);

    // suppress our tokenizer's stderr chatter
    std::freopen("nul", "w", stderr);

    QwenTokenizerV2 tokenizer;
    if (!tokenizer.load("../Qwen3_0.6B/vocab.json", "../Qwen3_0.6B/merges.txt")) {
        fprintf(stderr, "load failed\n");
        return 1;
    }

    // Register special tokens matching HF tokenizer.json
    tokenizer.add_special_token("<|endoftext|>", 151643);
    tokenizer.add_special_token("<|im_start|>", 151644);
    tokenizer.add_special_token("<|im_end|>", 151645);
    tokenizer.add_special_token("<|object_ref_start|>", 151646);
    tokenizer.add_special_token("<|object_ref_end|>", 151647);
    tokenizer.add_special_token("<|box_start|>", 151648);
    tokenizer.add_special_token("<|box_end|>", 151649);
    tokenizer.add_special_token("<|quad_start|>", 151650);
    tokenizer.add_special_token("<|quad_end|>", 151651);
    tokenizer.add_special_token("<|vision_start|>", 151652);
    tokenizer.add_special_token("<|vision_end|>", 151653);
    tokenizer.add_special_token("<|vision_pad|>", 151654);
    tokenizer.add_special_token("<|image_pad|>", 151655);
    tokenizer.add_special_token("<|video_pad|>", 151656);
    tokenizer.add_special_token("<tool_call>", 151657);
    tokenizer.add_special_token("</tool_call>", 151658);
    tokenizer.add_special_token("<|fim_prefix|>", 151659);
    tokenizer.add_special_token("<|fim_middle|>", 151660);
    tokenizer.add_special_token("<|fim_suffix|>", 151661);
    tokenizer.add_special_token("<|fim_pad|>", 151662);
    tokenizer.add_special_token("<|repo_name|>", 151663);
    tokenizer.add_special_token("<|file_sep|>", 151664);
    tokenizer.add_special_token("<tool_response>", 151665);
    tokenizer.add_special_token("</tool_response>", 151666);
    tokenizer.add_special_token("<think>", 151667);
    tokenizer.add_special_token("</think>", 151668);

    // warmup
    for (int i = 0; i < bi.warmup; ++i) {
        auto ids = tokenizer.encode(bi.text);
        tokenizer.decode(ids);
    }

    // encode benchmark
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<int> ids;
    for (int i = 0; i < bi.iterations; ++i) {
        ids = tokenizer.encode(bi.text);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double encode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bi.iterations;

    // decode benchmark
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bi.iterations; ++i) {
        tokenizer.decode(ids);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bi.iterations;

    // output results
    printf("ENCODE_MS=%.6f\n", encode_ms);
    printf("DECODE_MS=%.6f\n", decode_ms);
    printf("IDS=");
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) putchar(',');
        printf("%d", ids[i]);
    }
    putchar('\n');
    return 0;
}
