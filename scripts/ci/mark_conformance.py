#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Mark conformance ID as pass")
    parser.add_argument("--id", required=True, help="Conformance ID")
    args = parser.parse_args()

    out_dir = Path(".ci-results/conformance")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.id}.pass").write_text("PASS\n", encoding="utf-8")
    print(f"Marked {args.id} as PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
