#!/usr/bin/env python3
"""Compare token outputs between Run A and Run B by request ID (order-independent)."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 5:
        sys.exit("usage: compare_runs.py <run_a_dir> <run_b_dir> <model_tag> <report_out.json>")

    run_a = Path(sys.argv[1])
    run_b = Path(sys.argv[2])
    tag = sys.argv[3]
    report_path = Path(sys.argv[4])

    tokens_a = json.loads((run_a / "observables" / "tokens.json").read_text())
    tokens_b = json.loads((run_b / "observables" / "tokens.json").read_text())

    a_by_id = {r["id"]: r["tokens"] for r in tokens_a}
    b_by_id = {r["id"]: r["tokens"] for r in tokens_b}

    if set(a_by_id) != set(b_by_id):
        sys.exit(f"FAIL: request id sets differ. only-in-A={set(a_by_id)-set(b_by_id)} only-in-B={set(b_by_id)-set(a_by_id)}")

    matches = 0
    mismatches = []
    total_tokens = 0
    for rid in sorted(a_by_id):
        ta, tb = a_by_id[rid], b_by_id[rid]
        total_tokens += len(ta)
        if ta == tb:
            matches += 1
        else:
            first_diff = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
            mismatches.append({"id": rid, "first_diff_pos": first_diff, "len_a": len(ta), "len_b": len(tb)})

    report = {
        "model": tag,
        "test": "batch_order_invariance_1m",
        "total_requests": len(a_by_id),
        "token_matches": matches,
        "token_mismatches": len(mismatches),
        "total_tokens_compared": total_tokens,
        "status": "PASS" if not mismatches else "FAIL",
        "mismatch_details": mismatches[:20],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    if mismatches:
        print(f"FAIL: {len(mismatches)}/{len(a_by_id)} requests mismatch")
        for m in mismatches[:5]:
            print(f"  {m['id']}: diff@{m['first_diff_pos']} len_a={m['len_a']} len_b={m['len_b']}")
        sys.exit(1)
    print(f"PASS: {matches}/{len(a_by_id)} requests match, {total_tokens} total tokens identical")


if __name__ == "__main__":
    main()
