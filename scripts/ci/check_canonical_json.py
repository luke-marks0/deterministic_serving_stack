#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def canonical_text(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def iter_json_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.json") if p.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Path not found: {path}")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Check canonical JSON formatting")
    parser.add_argument("paths", nargs="+", help="JSON files or directories")
    parser.add_argument("--write", action="store_true", help="Rewrite files to canonical format")
    args = parser.parse_args()

    try:
        files = iter_json_files(args.paths)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not files:
        print("No JSON files found", file=sys.stderr)
        return 1

    bad: list[Path] = []
    for path in files:
        current = path.read_text(encoding="utf-8")
        try:
            data = json.loads(current)
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON: {path}: {exc}", file=sys.stderr)
            return 1

        canonical = canonical_text(data)
        if current != canonical:
            bad.append(path)
            if args.write:
                path.write_text(canonical, encoding="utf-8")

    if bad and not args.write:
        print("Non-canonical JSON files found:", file=sys.stderr)
        for item in bad:
            print(f"  - {item}", file=sys.stderr)
        return 1

    if bad and args.write:
        print(f"Rewrote {len(bad)} JSON file(s) to canonical formatting")
    else:
        print(f"All {len(files)} JSON file(s) are canonical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
