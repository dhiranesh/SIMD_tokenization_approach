#include "tokenizer_v2.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <cstdint>

static uint64_t fnv1a(const int* data, size_t n) {
    uint64_t h = 0xCBF29CE484222325ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t)(uint32_t)data[i];
        h *= 0x100000001B3ULL;
    }
    return h;
}

static bool read_file(const char* path, std::string& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    if (sz < 0) { std::fclose(f); return false; }
    std::fseek(f, 0, SEEK_SET);
    out.resize((size_t)sz);
    if (sz > 0 && std::fread(&out[0], 1, (size_t)sz, f) != (size_t)sz) {
        std::fclose(f); return false;
    }
    std::fclose(f);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: validate_binary <file1.md> [file2.md ...]\n");
        return 1;
    }

    std::freopen("nul", "w", stderr);

    QwenTokenizerV2 tokenizer;
    if (!tokenizer.load("../Qwen3_0.6B/vocab.json", "../Qwen3_0.6B/merges.txt")) {
        fprintf(stderr, "load failed\n");
        return 1;
    }

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

    for (int a = 1; a < argc; ++a) {
        std::string text;
        if (!read_file(argv[a], text)) {
            printf("%s|ERROR|read_failed\n", argv[a]);
            continue;
        }

        auto ids = tokenizer.encode(text);
        std::string decoded = tokenizer.decode(ids);
        bool rt_ok = (decoded == text);
        uint64_t hash = fnv1a(ids.data(), ids.size());

        printf("%s|COUNT=%zu|HASH=%llu|RT=%s",
               argv[a], ids.size(), (unsigned long long)hash, rt_ok ? "OK" : "FAIL");

        if (ids.size() > 0) {
            printf("|FIRST=");
            int limit = (ids.size() < 5) ? (int)ids.size() : 5;
            for (int i = 0; i < limit; ++i) {
                if (i) putchar(',');
                printf("%d", ids[i]);
            }
            printf("|LAST=");
            int start = (int)ids.size() - 5;
            if (start < 0) start = 0;
            for (int i = start; i < (int)ids.size(); ++i) {
                if (i > start) putchar(',');
                printf("%d", ids[i]);
            }
        }
        printf("\n");
        fflush(stdout);
    }

    return 0;
}
