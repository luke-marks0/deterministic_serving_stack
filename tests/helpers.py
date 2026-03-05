from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n", encoding="utf-8")
