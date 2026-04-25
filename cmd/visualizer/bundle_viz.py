#!/usr/bin/env python3
"""Generate an HTML visualization of a run bundle (and optional verify report).

Usage:
    python3 cmd/visualizer/bundle_viz.py \
        --bundle /path/to/run_bundle.v1.json \
        [--bundle-dir /path/to/bundle/]    # for loading observable files
        [--compare /path/to/other_bundle.v1.json] \
        [--report /path/to/verify_report.json] \
        --out report.html
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _short(digest: str, n: int = 12) -> str:
    if ":" in digest:
        return digest.split(":", 1)[1][:n]
    return digest[:n]


def _status_color(status: str) -> str:
    if status == "conformant":
        return "#22c55e"
    if status.startswith("non_conformant"):
        return "#f59e0b"
    if status == "mismatch_outputs":
        return "#ef4444"
    return "#94a3b8"


def _badge(text: str, color: str) -> str:
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
        f'font-size:13px;font-weight:600;color:#fff;background:{color}">{text}</span>'
    )


def _digest_pill(digest: str) -> str:
    short = _short(digest)
    return (
        f'<code style="background:#1e293b;color:#38bdf8;padding:2px 6px;'
        f'border-radius:4px;font-size:12px" title="{digest}">{short}</code>'
    )


def render_html(
    bundle: dict[str, Any],
    bundle_dir: Path | None = None,
    compare: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
) -> str:
    env = bundle.get("environment_info", {})
    hw_conf = bundle.get("hardware_conformance", {})
    obs = bundle.get("observables", {})
    exec_ctx = bundle.get("execution_context", {})
    trace_meta = bundle.get("execution_trace_metadata", {})
    attestations = bundle.get("attestations", [])

    # Load observables if bundle_dir provided
    token_data = None
    if bundle_dir:
        tokens_path = bundle_dir / obs.get("tokens", {}).get("path", "")
        if tokens_path.exists():
            token_data = json.loads(tokens_path.read_text())

    # Comparison section
    comparison_html = ""
    if report:
        status = report.get("status", "unknown")
        sc = _status_color(status)
        env_diffs = report.get("environment_diffs", {})
        comparison_html = f"""
        <div class="card">
            <h2>Verification Result</h2>
            <div style="text-align:center;padding:20px">
                {_badge(status.upper().replace("_", " "), sc)}
            </div>
            <table>
                <tr><td>Baseline</td><td><code>{report.get('baseline_run_id','')}</code></td></tr>
                <tr><td>Candidate</td><td><code>{report.get('candidate_run_id','')}</code></td></tr>
                <tr><td>Runtime closure match</td><td>{'Yes' if env_diffs.get('runtime_closure_digest_equal') else 'No'}</td></tr>
                <tr><td>Hardware match</td><td>{'Yes' if env_diffs.get('hardware_fingerprint_equal') else 'No'}</td></tr>
                <tr><td>Version diffs</td><td>{len(env_diffs.get('version_diffs', []))} difference(s)</td></tr>
            </table>
            {"".join(f'<div class="check {'check-pass' if c['outcome']=='pass' else 'check-fail'}">{c["conformance_id"]}: {c["outcome"]} — {c["detail"]}</div>' for c in report.get('checks', []))}
        </div>
        """

    # Token preview
    token_preview_html = ""
    if token_data and isinstance(token_data, list):
        rows = ""
        for item in token_data[:8]:
            toks = item.get("tokens", [])
            tok_str = " ".join(str(t) for t in toks[:12])
            if len(toks) > 12:
                tok_str += f" ... ({len(toks)} total)"
            tok_hash = hashlib.sha256(json.dumps(toks).encode()).hexdigest()[:12]
            rows += f"<tr><td><code>{item.get('id','')}</code></td><td>{len(toks)}</td><td class='mono'>{tok_str}</td><td>{_digest_pill('sha256:' + tok_hash)}</td></tr>"
        token_preview_html = f"""
        <div class="card">
            <h2>Token Observables</h2>
            <table>
                <tr><th>Request</th><th>Count</th><th>Tokens (preview)</th><th>Hash</th></tr>
                {rows}
            </table>
        </div>
        """

    # Observable digests
    obs_rows = ""
    for name in ["tokens", "logits", "activations", "engine_trace", "network_egress"]:
        info = obs.get(name, {})
        digest = info.get("digest", "")
        path = info.get("path", "")
        match_icon = ""
        if compare:
            other_digest = compare.get("observables", {}).get(name, {}).get("digest", "")
            if digest and other_digest:
                match_icon = "✓" if digest == other_digest else "✗"
        obs_rows += f"<tr><td><strong>{name}</strong></td><td class='mono'>{path}</td><td>{_digest_pill(digest) if digest else '—'}</td><td style='text-align:center'>{match_icon}</td></tr>"

    # Attestation rows
    att_rows = ""
    for att in attestations:
        att_rows += f"<tr><td>{att.get('attestation_type','')}</td><td><code>{att.get('signer','')}</code></td><td>{_digest_pill(att.get('statement_digest',''))}</td><td>{att.get('timestamp','')}</td></tr>"

    # Artifact digest summary
    artifacts = bundle.get("resolved_artifact_digests", [])
    art_rows = ""
    for art in artifacts[:12]:
        art_rows += f"<tr><td><code>{art.get('artifact_id','')}</code></td><td>{art.get('artifact_type','')}</td><td>{_digest_pill(art.get('digest',''))}</td></tr>"
    if len(artifacts) > 12:
        art_rows += f"<tr><td colspan=3 style='color:#64748b'>... and {len(artifacts)-12} more</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Run Bundle: {bundle.get('run_id','')}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#0f172a; color:#e2e8f0; padding:24px; max-width:960px; margin:0 auto; }}
  h1 {{ font-size:24px; margin-bottom:4px; }}
  h2 {{ font-size:16px; color:#94a3b8; margin-bottom:12px; border-bottom:1px solid #1e293b; padding-bottom:8px; }}
  .subtitle {{ color:#64748b; font-size:14px; margin-bottom:24px; }}
  .card {{ background:#1e293b; border-radius:8px; padding:20px; margin-bottom:16px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  @media (max-width:640px) {{ .grid {{ grid-template-columns:1fr; }} }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ text-align:left; color:#64748b; font-weight:500; padding:6px 8px; border-bottom:1px solid #334155; }}
  td {{ padding:6px 8px; border-bottom:1px solid #1e293b; vertical-align:top; }}
  code {{ font-family:'SF Mono',Consolas,monospace; font-size:12px; }}
  .mono {{ font-family:'SF Mono',Consolas,monospace; font-size:11px; color:#94a3b8; word-break:break-all; }}
  .digest-box {{ background:#0f172a; border:1px solid #334155; border-radius:6px; padding:12px; margin:8px 0; font-family:monospace; font-size:13px; color:#38bdf8; word-break:break-all; }}
  .chain {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin:12px 0; }}
  .chain-node {{ background:#334155; padding:6px 12px; border-radius:6px; font-size:12px; }}
  .chain-arrow {{ color:#64748b; font-size:18px; }}
  .check {{ padding:4px 8px; margin:2px 0; border-radius:4px; font-size:12px; }}
  .check-pass {{ background:#166534; color:#bbf7d0; }}
  .check-fail {{ background:#991b1b; color:#fecaca; }}
  .section-label {{ color:#64748b; font-size:11px; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; }}
</style>
</head>
<body>

<h1>Run Bundle</h1>
<div class="subtitle">{bundle.get('run_id','')} &middot; {bundle.get('created_at','')}</div>

<!-- Provenance Chain -->
<div class="card">
    <h2>Provenance Chain</h2>
    <div class="chain">
        <div class="chain-node">Manifest {_digest_pill(bundle.get('manifest_copy',{}).get('digest',''))}</div>
        <div class="chain-arrow">&rarr;</div>
        <div class="chain-node">Lockfile {_digest_pill(bundle.get('lockfile_copy',{}).get('digest',''))}</div>
        <div class="chain-arrow">&rarr;</div>
        <div class="chain-node">Closure {_digest_pill(bundle.get('runtime_closure_digest',''))}</div>
        <div class="chain-arrow">&rarr;</div>
        <div class="chain-node">Bundle {_digest_pill(bundle.get('bundle_digest',''))}</div>
    </div>
</div>

{comparison_html}

<!-- Environment + Hardware -->
<div class="grid">
    <div class="card">
        <h2>Environment</h2>
        <table>
            <tr><td>GPU</td><td>{', '.join(env.get('gpu_inventory', ['—']))}</td></tr>
            <tr><td>vLLM</td><td><code>{env.get('vllm_version','—')}</code></td></tr>
            <tr><td>PyTorch</td><td><code>{env.get('torch_version','—')}</code></td></tr>
            <tr><td>CUDA</td><td><code>{env.get('cuda_version','—')}</code></td></tr>
            <tr><td>Driver</td><td><code>{env.get('driver_version','—')}</code></td></tr>
            <tr><td>HW fingerprint</td><td>{_digest_pill(env.get('hardware_fingerprint',''))}</td></tr>
        </table>
    </div>
    <div class="card">
        <h2>Hardware Conformance</h2>
        <table>
            <tr><td>Status</td><td>{_badge(hw_conf.get('status','unknown'), _status_color(hw_conf.get('status','')))}</td></tr>
            <tr><td>Strict</td><td>{'Yes' if hw_conf.get('strict_hardware') else 'No'}</td></tr>
            <tr><td>Expected</td><td>{_digest_pill(hw_conf.get('expected_fingerprint',''))}</td></tr>
            <tr><td>Actual</td><td>{_digest_pill(hw_conf.get('actual_fingerprint',''))}</td></tr>
            <tr><td>Diffs</td><td>{len(hw_conf.get('diffs', []))} difference(s)</td></tr>
        </table>
    </div>
</div>

<!-- Observables -->
<div class="card">
    <h2>Observables {'(compared with ' + compare.get('run_id','') + ')' if compare else ''}</h2>
    <table>
        <tr><th>Observable</th><th>Path</th><th>Digest</th><th>{'Match' if compare else ''}</th></tr>
        {obs_rows}
    </table>
</div>

{token_preview_html}

<!-- Network Provenance -->
<div class="card">
    <h2>Network Provenance</h2>
    <table>
        <tr><td>Route mode</td><td><code>{net.get('route_mode','')}</code></td></tr>
        <tr><td>Security</td><td><code>{net.get('security_mode','')}</code></td></tr>
        <tr><td>Frames</td><td>{net.get('frame_count', 0)}</td></tr>
    </table>
</div>

<!-- Execution -->
<div class="card">
    <h2>Execution Context</h2>
    <table>
        <tr><td>Entrypoint</td><td><code>{exec_ctx.get('entrypoint','')}</code></td></tr>
        <tr><td>Replica</td><td><code>{exec_ctx.get('replica_id','')}</code></td></tr>
    </table>
</div>

<!-- Artifacts -->
<div class="card">
    <h2>Pinned Artifacts ({len(artifacts)})</h2>
    <table>
        <tr><th>ID</th><th>Type</th><th>Digest</th></tr>
        {art_rows}
    </table>
</div>

<!-- Attestations -->
<div class="card">
    <h2>Attestations</h2>
    <table>
        <tr><th>Type</th><th>Signer</th><th>Statement</th><th>Time</th></tr>
        {att_rows}
    </table>
</div>

<div style="text-align:center;color:#475569;font-size:11px;margin-top:24px">
    deterministic-serving-stack &middot; bundle visualizer
</div>
</body>
</html>"""
    return html


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bundle HTML visualizer")
    parser.add_argument("--bundle", required=True, help="Path to run_bundle.v1.json")
    parser.add_argument("--bundle-dir", help="Bundle directory (for loading observable files)")
    parser.add_argument("--compare", help="Second bundle for side-by-side comparison")
    parser.add_argument("--report", help="Verify report JSON")
    parser.add_argument("--out", required=True, help="Output HTML path")
    args = parser.parse_args()

    bundle = _load(Path(args.bundle))
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else Path(args.bundle).parent
    compare = _load(Path(args.compare)) if args.compare else None
    report = _load(Path(args.report)) if args.report else None

    html = render_html(bundle, bundle_dir, compare, report)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"Written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
