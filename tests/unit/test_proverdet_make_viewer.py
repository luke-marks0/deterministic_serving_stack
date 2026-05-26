"""Unit test for make_viewer.py (Task 9.3).

Asserts that the generator runs on a synthetic results.jsonl and produces a
single, self-contained HTML file under the size budget. No browser-side
checks (the plan accepts skipping the playwright smoke test if it's not
already a dependency — it isn't here).
"""

from __future__ import annotations

import json
import re
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
    observed: int,
    claimed: int,
    traffic: int,
) -> dict[str, object]:
    return {
        "workload": workload,
        "knob_param": knob_param,
        "knob_value": knob_value,
        "verdict": verdict,
        "reasons": [],
        "signals": signals,
        "observed_flops": observed,
        "claimed_flops": claimed,
        "traffic_size": traffic,
        "runtime_s": 0.1,
    }


class TestMakeViewer(unittest.TestCase):
    def test_make_viewer_writes_self_contained_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results = tmp_path / "results.jsonl"
            out = tmp_path / "viewer.html"

            rows = [
                _row(
                    "benign",
                    "seed",
                    0,
                    verdict="inference",
                    signals={
                        "replay_correctness": "pass",
                        "compute_budget": "pass",
                        "bandwidth": "pass",
                    },
                    observed=5120,
                    claimed=5120,
                    traffic=5120,
                ),
                _row(
                    "mixed_lora",
                    "gradient_steps",
                    8,
                    verdict="training_or_exfil",
                    signals={
                        "replay_correctness": "pass",
                        "compute_budget": "fail",
                        "bandwidth": "pass",
                    },
                    observed=20_000,
                    claimed=5120,
                    traffic=5120,
                ),
                _row(
                    "lora_loading",
                    "lora_bytes",
                    524_288,
                    verdict="training_or_exfil",
                    signals={
                        "replay_correctness": "pass",
                        "compute_budget": "pass",
                        "bandwidth": "fail",
                    },
                    observed=2560,
                    claimed=2560,
                    traffic=526_848,
                ),
            ]
            results.write_text(
                "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
                encoding="utf-8",
            )

            rc = subprocess.run(
                [
                    sys.executable,
                    "demos/prover-verifier/scripts/make_viewer.py",
                    "--results",
                    str(results),
                    "--out",
                    str(out),
                ],
                cwd=str(REPO_ROOT),
                env=sandbox_env(),
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(rc.returncode, 0, f"stderr={rc.stderr}")
            self.assertTrue(out.exists())

            html = out.read_text(encoding="utf-8")
            # Self-contained: no external script/stylesheet references.
            self.assertNotRegex(html, r"<script\s+[^>]*src=", "external script tag found")
            self.assertNotRegex(
                html, r"<link\s+[^>]*rel=['\"]stylesheet", "external CSS link found"
            )
            self.assertNotIn("cdn.", html.lower())
            self.assertNotIn("https://", html)  # no external URL refs
            # Embedded payload reachable.
            self.assertIn("benign", html)
            self.assertIn("mixed_lora", html)
            self.assertIn("lora_loading", html)
            self.assertIn("training_or_exfil", html)
            self.assertIn("inference", html)
            # Inline SVG curve (the headline figure — recomputed in JS).
            self.assertIn("<svg", html)
            # Plan budget: under 50 KB.
            self.assertLess(
                out.stat().st_size, 50 * 1024, f"viewer.html too big: {out.stat().st_size}"
            )
            # Sanity: payload is valid JSON.
            m = re.search(r"const\s+BUNDLE\s*=\s*(\{.+?\});", html, re.DOTALL)
            self.assertIsNotNone(m, "could not find BUNDLE constant")
            assert m is not None
            payload = json.loads(m.group(1))
            self.assertIn("rows", payload)
            self.assertEqual(len(payload["rows"]), 3)


if __name__ == "__main__":
    unittest.main()
