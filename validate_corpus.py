"""
validate_corpus.py — Validate C++ tokenizers against HF reference on the 7K MD corpus.

Runs validate_base.exe and validate_fixed2.exe on all .md files, compares
token IDs against Hugging Face tokenizers library, and produces a report.
"""
import os, sys, subprocess, time, glob, json, struct, tempfile
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CORPUS_ROOT = r"G:\Gpu inferance_sheduler\Segregated_7k_by_DocType"
TOKENIZER_DIR = r"D:\GPU inferance_tool\tokenizer_v2"
HF_TOKENIZER_JSON = r"D:\GPU inferance_tool\Qwen3_0.6B\tokenizer.json"

def find_md_files(root):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith('.md'):
                files.append(os.path.join(dirpath, fn))
    return sorted(files)

def fnv1a_hash(ids):
    h = 0xCBF29CE484222325
    for idv in ids:
        h ^= idv & 0xFFFFFFFF
        h *= 0x100000001B3
        h &= 0xFFFFFFFFFFFFFFFF
    return h

def parse_validate_output(output):
    results = {}
    for line in output.strip().split('\n'):
        line = line.strip()
        if '|' not in line: continue
        parts = line.split('|')
        fpath = parts[0]
        info = {}
        for p in parts[1:]:
            if '=' in p:
                k, v = p.split('=', 1)
                info[k] = v
        results[fpath] = info
    return results

def main():
    print("=" * 60)
    print("  Corpus Validation: C++ SIMD Tokenizer vs HF Reference")
    print("=" * 60)
    
    # Find all MD files
    print("\n[1] Finding .md files...")
    files = find_md_files(CORPUS_ROOT)
    print(f"    Found {len(files)} files")
    
    # Load HF tokenizer (once)
    print("\n[2] Loading HF tokenizer...")
    from tokenizers import Tokenizer
    hf = Tokenizer.from_file(HF_TOKENIZER_JSON)
    print("    OK")
    
    # Compute HF IDs for each file (with caching)
    hf_cache_path = os.path.join(TOKENIZER_DIR, "hf_cache.json")
    hf_results = {}
    
    # Try loading cache
    if os.path.exists(hf_cache_path):
        print("\n[3] Loading HF reference from cache...")
        with open(hf_cache_path, 'r', encoding='utf-8') as f:
            raw_cache = json.load(f)
        # Convert string keys back, and 'ids' from list
        for k, v in raw_cache.items():
            v['ids'] = [int(x) for x in v.get('ids_raw', '').split(',')] if v.get('ids_raw') else []
            v['first5'] = [int(x) for x in v.get('first5_raw', '').split(',')] if v.get('first5_raw') else []
            v['last5'] = [int(x) for x in v.get('last5_raw', '').split(',')] if v.get('last5_raw') else []
            hf_results[k] = v
        print(f"    Loaded {len(hf_results)} cached results")
    
    # Fill in missing files
    missing = [f for f in files if f not in hf_results]
    if missing:
        print(f"\n    Computing HF for {len(missing)} uncached files...")
        t0 = time.time()
        for i, fpath in enumerate(missing):
            try:
                with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
                enc = hf.encode(text)
                ids = enc.ids
                hf_results[fpath] = {
                    'count': len(ids),
                    'hash': fnv1a_hash(ids),
                    'first5': ids[:5],
                    'last5': ids[-5:] if ids else [],
                    'ids': ids,
                }
            except Exception as e:
                hf_results[fpath] = {'error': str(e)}
            if (i + 1) % 200 == 0 or i == len(missing) - 1:
                elapsed = time.time() - t0
                print(f"    [{i+1:4d}/{len(missing)}] {elapsed:.1f}s")
        print(f"    HF done in {time.time()-t0:.1f}s")
        
        # Save cache
        cache_out = {}
        for k, v in hf_results.items():
            entry = dict(v)
            entry['ids_raw'] = ','.join(str(x) for x in v.get('ids', []))
            entry.pop('ids', None)
            entry['first5_raw'] = ','.join(str(x) for x in v.get('first5', []))
            entry.pop('first5', None)
            entry['last5_raw'] = ','.join(str(x) for x in v.get('last5', []))
            entry.pop('last5', None)
            cache_out[k] = entry
        with open(hf_cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_out, f, ensure_ascii=False)
        print(f"    Cached {len(cache_out)} results")
    
    def run_validator(exe_name, files_batch):
        """Run validator in batches to avoid Windows command-line length limit."""
        validate_path = os.path.join(TOKENIZER_DIR, exe_name)
        if not os.path.exists(validate_path):
            print(f"    ERROR: {validate_path} not found")
            return {}
        batch_size = 30
        results = {}
        t0 = time.time()
        for i in range(0, len(files_batch), batch_size):
            batch = files_batch[i:i+batch_size]
            r = subprocess.run([validate_path] + batch, capture_output=True, timeout=120)
            stdout_text = r.stdout.decode('utf-8', errors='replace')
            batch_results = parse_validate_output(stdout_text)
            results.update(batch_results)
            if (i + batch_size) % 120 == 0 or i + batch_size >= len(files_batch):
                print(f"    [{i + len(batch):4d}/{len(files_batch)}] {time.time()-t0:.1f}s")
        print(f"    Done in {time.time()-t0:.1f}s, {len(results)} results")
        return results
    
    # Run validate_base.exe
    print("\n[4] Running validate_base.exe...")
    base_results = run_validator("validate_base.exe", files)
    
    # Run validate_fixed2.exe
    print("\n[5] Running validate_fixed2.exe...")
    fixed2_results = run_validator("validate_fixed2.exe", files)
    
    # Compare
    print("\n[6] Comparing results...")
    base_ok = 0
    base_fail = 0
    fixed2_ok = 0
    fixed2_fail = 0
    base_mismatches = []
    fixed2_mismatches = []
    rt_fail_base = 0
    rt_fail_fixed2 = 0
    
    for fpath in files:
        hf = hf_results.get(fpath, {})
        base = base_results.get(fpath, {})
        fixed2 = fixed2_results.get(fpath, {})
        
        if 'error' in hf:
            continue
        
        # Base comparison
        base_count = base.get('COUNT', '')
        base_hash = base.get('HASH', '')
        base_rt = base.get('RT', '')
        try:
            hf_count = hf['count']
            hf_hash = hf['hash']
            if base_count and int(base_count) == hf_count and base_hash and int(base_hash) == hf_hash:
                base_ok += 1
            else:
                base_fail += 1
                base_mismatches.append((fpath, hf_count, base_count, hf_hash, base_hash))
        except (ValueError, KeyError):
            base_fail += 1
            base_mismatches.append((fpath, hf.get('count'), base.get('COUNT'), hf.get('hash'), base.get('HASH')))
        
        if base_rt == 'FAIL':
            rt_fail_base += 1
        
        # Fixed2 comparison
        f2_count = fixed2.get('COUNT', '')
        f2_hash = fixed2.get('HASH', '')
        f2_rt = fixed2.get('RT', '')
        try:
            if f2_count and int(f2_count) == hf_count and f2_hash and int(f2_hash) == hf_hash:
                fixed2_ok += 1
            else:
                fixed2_fail += 1
                fixed2_mismatches.append((fpath, hf_count, f2_count, hf_hash, f2_hash))
        except (ValueError, KeyError):
            fixed2_fail += 1
            fixed2_mismatches.append((fpath, hf.get('count'), fixed2.get('COUNT'), hf.get('hash'), fixed2.get('HASH')))
        
        if f2_rt == 'FAIL':
            rt_fail_fixed2 += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("  VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total files: {len(files)}")
    print(f"\n  validate_base.exe (base sources):")
    print(f"    Match: {base_ok}/{len(files)} ({100*base_ok/len(files):.1f}%)")
    print(f"    Mismatch: {base_fail}/{len(files)} ({100*base_fail/len(files):.1f}%)")
    print(f"    Roundtrip failures: {rt_fail_base}")
    
    if base_mismatches:
        print(f"\n  Base mismatches (first 10):")
        for fpath, hfc, bc, hfh, bh in base_mismatches[:10]:
            fname = os.path.basename(fpath)
            print(f"    {fname}: HF={hfc} ours={bc} hash={hfh} vs {bh}")
    
    print(f"\n  validate_fixed2.exe (fixed2 sources):")
    print(f"    Match: {fixed2_ok}/{len(files)} ({100*fixed2_ok/len(files):.1f}%)")
    print(f"    Mismatch: {fixed2_fail}/{len(files)} ({100*fixed2_fail/len(files):.1f}%)")
    print(f"    Roundtrip failures: {rt_fail_fixed2}")
    
    if fixed2_mismatches:
        print(f"\n  Fixed2 mismatches (first 10):")
        for fpath, hfc, f2c, hfh, f2h in fixed2_mismatches[:10]:
            fname = os.path.basename(fpath)
            tok_diff = ""
            if hfc and f2c:
                try:
                    tok_diff = f" delta={int(f2c)-int(hfc):+d}"
                except ValueError:
                    pass
            print(f"    {fname}: HF={hfc} ours={f2c}{tok_diff}")
    
    # Detailed mismatch report for fixed2
    print(f"\n{'='*60}")
    print("  FIXED2 MISMATCH ANALYSIS")
    print(f"{'='*60}")
    if fixed2_mismatches:
        # Show size distribution of mismatched files
        sizes = []
        for fpath, hfc, f2c, _, _ in fixed2_mismatches:
            if hfc and f2c:
                try:
                    sizes.append(('abs', abs(int(hfc) - int(f2c))))
                    sizes.append(('rel', abs(int(hfc) - int(f2c)) / max(int(hfc), 1) * 100))
                except ValueError:
                    pass
        
        if sizes:
            abs_deltas = [s[1] for s in sizes if s[0] == 'abs']
            rel_deltas = [s[1] for s in sizes if s[0] == 'rel']
            print(f"  Token count delta stats:")
            print(f"    Mean absolute delta: {sum(abs_deltas)/len(abs_deltas):.1f}")
            print(f"    Max absolute delta: {max(abs_deltas)}")
            print(f"    Mean relative delta: {sum(rel_deltas)/len(rel_deltas):.2f}%")
            print(f"    Max relative delta: {max(rel_deltas):.2f}%")
        
        # Find the worst-mismatched file for deep analysis
        worst = None
        worst_delta = 0
        for fpath, hfc, f2c, _, _ in fixed2_mismatches:
            if hfc and f2c:
                try:
                    d = abs(int(hfc) - int(f2c))
                    if d > worst_delta:
                        worst_delta = d
                        worst = fpath
                except ValueError:
                    pass
        
        if worst:
            print(f"\n  Deep analysis for worst file: {os.path.basename(worst)}")
            with open(worst, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            hf_ids = hf_results[worst]['ids']
            
            # Get fixed2 IDs for comparison
            r = subprocess.run([validate_fixed2, worst], capture_output=True, text=True, timeout=30)
            for line in r.stdout.strip().split('\n'):
                if '|' in line and worst.replace('\\', '/') in line.replace('\\', '/'):
                    parts = line.split('|')
                    for p in parts[1:]:
                        if p.startswith('COUNT='):
                            pass
            
            # Rerun fixed2 with ID output for this file
            # Use the bench_runner approach temporarily
            print(f"    HF count: {len(hf_ids)}")
            print(f"    Fixed2 count: {worst_delta} delta")
            
            # Find first position where fixed2 diverges
            # We need to get fixed2's actual IDs for this file
            # Use a temporary binary file with bench_runner
            # Actually validate_fixed2 doesn't output all IDs
            # Let's use the original qwen_v2_bench approach for deep dive
    
    print(f"\n{'='*60}")
    print(f"  Done. Base match rate: {100*base_ok/len(files):.1f}%")
    print(f"         Fixed2 match rate: {100*fixed2_ok/len(files):.1f}%")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
