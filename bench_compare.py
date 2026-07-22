"""
bench_compare.py — Structured performance comparison:
  HF Rust-backed tokenizers library  vs  Our C++ SIMD BPE tokenizer

Runs encode + decode on 3 size classes (small ~2KB, medium ~50KB, large ~450KB)
and reports throughput.  Verifies token-ID correctness against the HF reference.
"""
import json, os, sys, time, subprocess, tempfile, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HF_TOKENIZER_JSON = "../Qwen3_0.6B/tokenizer.json"
VOCAB_JSON = "../Qwen3_0.6B/vocab.json"
MERGES_TXT = "../Qwen3_0.6B/merges.txt"

OUR_EXE = "qwen_v2_bench.exe"
SRC_DIR = "."

# -- build our test binary (one-shot) ----------------------------------
def build_our_tokenizer():
    objs = []
    srcs = [
        "src/arena.cpp", "src/flat_vocab.cpp", "src/bpe_engine.cpp",
        "src/pre_tokenizer.cpp", "src/tokenizer_v2.cpp",
        "src/simd_dispatch.cpp", "src/simd_scalar.cpp", "src/simd_avx2.cpp",
    ]
    try:
        for s in srcs:
            o = s.replace("src/", "").replace(".cpp", ".o")
            r = subprocess.run(
                ["g++", "-std=c++17", "-O2", "-mavx2", "-mbmi2", "-c", s,
                 "-Iinclude", "-o", o],
                capture_output=True, text=True, cwd=SRC_DIR)
            if r.returncode != 0:
                print("BUILD FAIL:", r.stderr)
                return False
            objs.append(o)
        r = subprocess.run(
            ["g++", "-std=c++17", "-O2", "-mavx2", "-mbmi2",
             "-o", OUR_EXE, "bench_runner.cpp"] + objs + ["-Iinclude"],
            capture_output=True, text=True, cwd=SRC_DIR)
        if r.returncode != 0:
            print("BUILD FAIL (link):", r.stderr)
            return False
        return True
    except FileNotFoundError:
        print("g++ not found")
        return False


# -- HF benchmark ------------------------------------------------------
def bench_hf(text, n_warmup=3, n_iter=10):
    """Returns (encode_ms_per_1KB, decode_ms_per_1KB) for HF tokenizers lib."""
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(HF_TOKENIZER_JSON)

    text_bytes = len(text.encode("utf-8"))
    text_kb = text_bytes / 1024.0

    # warmup
    for _ in range(n_warmup):
        enc = t.encode(text)
        t.decode(enc.ids)

    # encode
    t0 = time.perf_counter()
    for _ in range(n_iter):
        enc = t.encode(text)
    encode_s = (time.perf_counter() - t0) / n_iter

    # decode
    ids = enc.ids
    t0 = time.perf_counter()
    for _ in range(n_iter):
        t.decode(ids)
    decode_s = (time.perf_counter() - t0) / n_iter

    return encode_s / text_kb * 1000, decode_s / text_kb * 1000, enc.ids


# -- our C++ benchmark -------------------------------------------------
def bench_our(text, n_warmup=3, n_iter=10):
    """Returns (encode_ms_per_1KB, decode_ms_per_1KB, ids) for our tokenizer."""
    import tempfile, struct
    text_bytes = text.encode("utf-8")
    text_kb = len(text_bytes) / 1024.0

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        fname = f.name
        f.write(struct.pack("<I", len(text_bytes)))
        f.write(text_bytes)
        f.write(struct.pack("<I", n_warmup))
        f.write(struct.pack("<I", n_iter))

    try:
        r = subprocess.run(["./" + OUR_EXE, fname],
                           capture_output=True, text=True, timeout=120,
                           cwd=SRC_DIR)
        if r.returncode != 0:
            print("STDERR:", r.stderr)
            raise RuntimeError(f"bench binary failed: {r.stderr}")

        # Parse output
        # Expected format:
        #   ENCODE_MS=<float>
        #   DECODE_MS=<float>
        #   IDS=<comma-sep ints>
        encode_ms = None
        decode_ms = None
        ids = []
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.startswith("ENCODE_MS="):
                encode_ms = float(line.split("=")[1])
            elif line.startswith("DECODE_MS="):
                decode_ms = float(line.split("=")[1])
            elif line.startswith("IDS="):
                ids = [int(x) for x in line.split("=", 1)[1].split(",") if x.strip()]

        if encode_ms is None or decode_ms is None:
            raise RuntimeError(f"Unexpected output: {r.stdout[:500]}")
        return encode_ms / text_kb, decode_ms / text_kb, ids
    finally:
        os.unlink(fname)


# -- build test text ---------------------------------------------------
def build_text(target_kb):
    """Build a realistic text of roughly target_kb kilobytes."""
    # Mix English sentences + some Chinese + numbers + special chars
    english = (
        "The Qwen3 tokenizer is a Byte Pair Encoding tokenizer with byte-level fallback. "
        "It supports multiple languages including Chinese, Japanese, Korean, and various "
        "European languages. The model is trained on a large corpus of diverse text. "
        "Tokenization is the first step in natural language processing pipelines. "
    )
    chinese = "Qwen3分词器是一种基于字节对编码的分词工具，支持多种语言。它在大规模语料库上进行了训练，能够高效地处理中文文本。"
    numbers = " 42 3.14159 2.71828 1000000 0xFF 0b1010 1234567890 1.5e-10 "
    special = " @#$%^&*()_+-=[]{}|;':\",./<>?`~ !@#$%^&*()_+-=[]{}|;':\",./<>?`~ "

    paragraph = english + chinese + numbers + special + "\n"
    reps = max(1, int(target_kb * 1024 / len(paragraph.encode("utf-8"))))
    result = (paragraph * reps)[:int(target_kb * 1024)]
    # ensure we end with a newline
    if not result.endswith("\n"):
        result += "\n"
    return result


# -- main --------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Structured Performance Comparison")
    print("  HF tokenizers (Rust) vs Our SIMD C++ BPE Tokenizer")
    print("=" * 60)

    # Build our binary
    print("\n[1] Building our C++ benchmark binary...")
    if not build_our_tokenizer():
        sys.exit(1)
    print("    OK")

    # Build test texts
    print("\n[2] Building test texts...")
    texts = {
        "small  (2 KB)":  build_text(2),
        "medium (50 KB)": build_text(50),
        "large  (450 KB)": build_text(450),
    }
    for label, t in texts.items():
        print(f"    {label}: {len(t.encode('utf-8')):,} bytes")

    # Run benchmarks
    print("\n[3] Running benchmarks...")
    results = []
    for label, text in texts.items():
        print(f"\n    -- {label} --")
        text_bytes = len(text.encode("utf-8"))
        text_kb = text_bytes / 1024.0

        n_iter = 50 if text_bytes < 20000 else (20 if text_bytes < 100000 else 10)

        # HF
        hf_enc_ms_kb, hf_dec_ms_kb, hf_ids = bench_hf(text, n_warmup=3, n_iter=n_iter)
        hf_enc_mbps = text_bytes / 1024 / 1024 / (hf_enc_ms_kb * text_kb / 1000)
        hf_dec_mbps = text_bytes / 1024 / 1024 / (hf_dec_ms_kb * text_kb / 1000)

        # Our
        our_enc_ms_kb, our_dec_ms_kb, our_ids = bench_our(text, n_warmup=3, n_iter=n_iter)
        our_enc_mbps = text_bytes / 1024 / 1024 / (our_enc_ms_kb * text_kb / 1000)
        our_dec_mbps = text_bytes / 1024 / 1024 / (our_dec_ms_kb * text_kb / 1000)

        # Correctness
        ids_match = hf_ids == our_ids
        n_tokens = len(hf_ids)

        results.append({
            "label": label,
            "text_kb": text_kb,
            "n_tokens": n_tokens,
            "hf_enc_ms_kb": hf_enc_ms_kb,
            "hf_dec_ms_kb": hf_dec_ms_kb,
            "hf_enc_mbps": hf_enc_mbps,
            "hf_dec_mbps": hf_dec_mbps,
            "our_enc_ms_kb": our_enc_ms_kb,
            "our_dec_ms_kb": our_dec_ms_kb,
            "our_enc_mbps": our_enc_mbps,
            "our_dec_mbps": our_dec_mbps,
            "ids_match": ids_match,
        })

        ratio_enc = our_enc_ms_kb / hf_enc_ms_kb if hf_enc_ms_kb else float('inf')
        ratio_dec = our_dec_ms_kb / hf_dec_ms_kb if hf_dec_ms_kb else float('inf')

        print(f"      {'HF':>20s}  {'Ours':>20s}  {'Ratio':>8s}")
        print(f"      {'-'*20}  {'-'*20}  {'-'*8}")
        print(f"      {'Encode (ms/KB):':>20s}  {hf_enc_ms_kb:>8.4f} ms       {our_enc_ms_kb:>8.4f} ms   {ratio_enc:>6.2f}x")
        print(f"      {'Encode (MB/s):':>20s}  {hf_enc_mbps:>8.2f}          {our_enc_mbps:>8.2f}")
        print(f"      {'Decode (ms/KB):':>20s}  {hf_dec_ms_kb:>8.4f} ms       {our_dec_ms_kb:>8.4f} ms   {ratio_dec:>6.2f}x")
        print(f"      {'Decode (MB/s):':>20s}  {hf_dec_mbps:>8.2f}          {our_dec_mbps:>8.2f}")
        print(f"      {'Tokens:':>20s}  {n_tokens}")
        print(f"      {'IDs match:':>20s}  {ids_match}")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Size':>15s}  {'HF enc':>10s}  {'Our enc':>10s}  {'HF dec':>10s}  {'Our dec':>10s}  {'Match?':>8s}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
    for r in results:
        print(f"  {r['label']:>15s}  {r['hf_enc_mbps']:>8.2f} MB/s  {r['our_enc_mbps']:>8.2f} MB/s  "
              f"{r['hf_dec_mbps']:>8.2f} MB/s  {r['our_dec_mbps']:>8.2f} MB/s  {str(r['ids_match']):>8s}")


if __name__ == "__main__":
    main()
