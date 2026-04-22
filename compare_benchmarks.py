"""
compare_benchmarks.py
---------------------
Side-by-side diff of two benchmark_harness JSON reports.

Usage:
    python compare_benchmarks.py a.json b.json

Prints, per (instance, num_workers):
  - mean_s for A and B
  - delta absolute (B - A) and delta percent
  - fingerprint match (✓ / MISMATCH)
"""

import json
import sys
from typing import Dict, List, Tuple


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _index(results: List[dict]) -> Dict[Tuple[str, int], dict]:
    return {(r.get("instance"), r.get("num_workers")): r for r in results}


def main():
    if len(sys.argv) != 3:
        print("usage: compare_benchmarks.py <before.json> <after.json>",
              file=sys.stderr)
        sys.exit(2)

    a = _load(sys.argv[1])
    b = _load(sys.argv[2])

    print(f"A: {sys.argv[1]}")
    print(f"   {a.get('gpu_name')}  git={a.get('git_sha')}"
          f"{'-dirty' if a.get('git_dirty') else ''}  tag={a.get('tag', '')}")
    print(f"B: {sys.argv[2]}")
    print(f"   {b.get('gpu_name')}  git={b.get('git_sha')}"
          f"{'-dirty' if b.get('git_dirty') else ''}  tag={b.get('tag', '')}")
    print()

    idx_a = _index(a["results"])
    idx_b = _index(b["results"])
    keys = sorted(set(idx_a) | set(idx_b))

    header = (f"{'instance':<12} {'w':>2}  {'A mean_s':>10} {'B mean_s':>10}"
              f"  {'Δs':>8}  {'Δ%':>7}   fp")
    print(header)
    print("-" * len(header))

    for key in keys:
        instance, nw = key
        ra = idx_a.get(key)
        rb = idx_b.get(key)
        if ra is None or rb is None:
            side = "A" if rb is None else "B"
            print(f"{instance:<12} {nw:>2}  (missing from {side})")
            continue
        if "error" in ra or "error" in rb:
            print(f"{instance:<12} {nw:>2}  ERROR  "
                  f"A={ra.get('error', '-')}  B={rb.get('error', '-')}")
            continue

        ma = ra["mean_s"]
        mb = rb["mean_s"]
        d = mb - ma
        pct = (d / ma) * 100.0 if ma > 0 else float("nan")
        fp_ok = ra["fingerprint"] == rb["fingerprint"]
        fp_mark = "✓" if fp_ok else "MISMATCH"
        print(f"{instance:<12} {nw:>2}  {ma:>10.3f} {mb:>10.3f}  "
              f"{d:>+8.3f}  {pct:>+6.1f}%   {fp_mark}")


if __name__ == "__main__":
    main()
