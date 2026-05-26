"""Unit test for plot_results.py (Task 9.2).

Builds a synthetic results.jsonl (no real prover/verifier needed), invokes
the plot script, and asserts that the expected PNG files exist and are
non-empty. We don't pixel-match — that's brittle.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.proverdet._helpers import REPO_ROOT, sandbox_env


def _row(
    workload: str,
    knob_param: str,
    knob_value: int,
    *,
    verdict: str,
    signals: dict[str, str],
) -> dict[str, object]:
    return {
        "workload": workload,
        "knob_param": knob_param,
        "knob_value": knob_value,
        "verdict": verdict,
        "reasons": [],
        "signals": signals,
        "observed_flops": 0,
        "claimed_flops": 0,
        "traffic_size": 0,
        "runtime_s": 0.1,
    }


class TestPlotResults(unittest.TestCase):
    def test_plot_writes_two_pngs_for_synthetic_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results = tmp_path / "results.jsonl"
            figures_dir = tmp_path / "figures"

            rows: list[dict[str, object]] = [
                _row(
                    "mixed_lora",
                    "gradient_steps",
                    g,
                    verdict="inference" if g == 0 else "training_or_exfil",
                    signals={
                        "replay_correctness": "pass",
                        "compute_budget": "pass" if g == 0 else "fail",
                        "bandwidth": "pass",
                    },
                )
                for g in (0, 1, 2, 4, 8, 16)
            ]
            rows += [
                _row(
                    "lora_loading",
                    "lora_bytes",
                    b,
                    verdict="inference" if b == 0 else "training_or_exfil",
                    signals={
                        "replay_correctness": "pass",
                        "compute_budget": "pass",
                        "bandwidth": "pass" if b == 0 else "fail",
                    },
                )
                for b in (0, 4096, 65_536, 262_144, 1_048_576)
            ]
            results.write_text(
                "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
                encoding="utf-8",
            )

            rc = subprocess.run(
                [
                    sys.executable,
                    "demos/prover-verifier/scripts/plot_results.py",
                    "--results",
                    str(results),
                    "--figures-dir",
                    str(figures_dir),
                ],
                cwd=str(REPO_ROOT),
                env=sandbox_env(),
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(rc.returncode, 0, f"stderr={rc.stderr}")

            mixed = figures_dir / "mixed_lora_detection.png"
            lora = figures_dir / "lora_loading_detection.png"
            self.assertTrue(mixed.exists(), f"missing {mixed}")
            self.assertTrue(lora.exists(), f"missing {lora}")
            # Sanity: PNG header + non-trivial size.
            self.assertGreater(mixed.stat().st_size, 1024)
            self.assertGreater(lora.stat().st_size, 1024)
            self.assertEqual(mixed.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(lora.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
