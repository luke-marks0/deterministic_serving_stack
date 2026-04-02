# Manifest v2 Implementation Plan

## Background

The manifest is a JSON document that fully declares an inference workload: which model, which hardware, which settings, which requests. The server receives a manifest and configures vLLM to execute it.

The current manifest schema (`schemas/manifest.v1.schema.json`) has accumulated fields that are either redundant, fabricated, or never enforced. This plan cleans it up so that **every field in the manifest is either passed to vLLM, verified against the runtime environment, or used by downstream tools (verifier, capture)**. Nothing decorative remains.

### What you're working with

- **Language:** Python 3.10+ (server runs in a nix-built container with Python 3.12)
- **Test framework:** `unittest` (no pytest in CI; pytest works locally but CI uses `python3 -m unittest discover`)
- **CI gates:** `make ci-pr` runs lint + schema validation + unit/integration tests. All must pass.
- **Lint:** `scripts/ci/lint.sh` — checks for merge-conflict markers (`=======` at start of line — don't use `=======` in docstrings!) and `py_compile` on all `.py` files. No linter like ruff/flake8.
- **Schema validation:** `scripts/ci/schema_gate.sh` — validates all fixtures against schemas, checks canonical JSON, validates lockfile.
- **Current test counts:** 123 unit tests, 19 integration tests, all passing.

### How to run tests

```bash
# Unit tests only (fast, no GPU)
python3 -m unittest discover -s tests/unit -p 'test_*.py' -v

# Integration tests (no GPU needed for most)
python3 -m unittest discover -s tests/integration -p 'test_*.py' -v

# Full CI gate (lint + schema + tests)
make ci-pr

# Single test file
python3 -m unittest tests/unit/test_manifest_endpoint -v
```

### Key files

| File | What it does |
|------|-------------|
| `schemas/manifest.v1.schema.json` | JSON Schema for the manifest. Single line of minified JSON. |
| `schemas/lockfile.v1.schema.json` | JSON Schema for the lockfile. **Also needs updating** — has `allOf[contains]` constraints requiring artifact types we're removing. |
| `schemas/run_bundle.v1.schema.json` | JSON Schema for run bundles. **Also needs updating** — requires `activations`, `engine_trace`, `network_egress`, `actual_batch_sizes`, `network_provenance`. |
| `schemas/verify_report.v1.schema.json` | JSON Schema for verification reports. **Also needs updating** — has `network_trace_diffs`, `batch_trace_diffs`, `non_conformant_network` status. |
| `manifests/qwen3-1.7b.manifest.json` | Real manifest used in experiments. Must stay valid. |
| `tests/fixtures/positive/*.json` | Example manifests/lockfiles/bundles used by CI fixture validation. Must stay valid. |
| `tests/fixtures/negative/*.json` | Invalid documents that must FAIL validation. |
| `cmd/server/main.py` | The proxy server with POST/GET /manifest endpoint. |
| `cmd/resolver/main.py` | Resolves authored manifests → pinned manifests + lockfiles. |
| `pkg/common/hf_resolution.py` | HuggingFace model resolution logic. Line 499 reads `requested_revision` / `resolved_revision` — must update. |
| `cmd/runner/main.py` | Generates run bundles (synthetic mode, no GPU). Calls `create_net_stack()`. |
| `cmd/runner/vllm_runner.py` | Generates run bundles (real vLLM mode, needs GPU). |
| `cmd/capture/main.py` | Converts server capture logs to run bundles. Reads `network`, `batch_cardinality`. |
| `cmd/verifier/main.py` | Compares two run bundles. Has extensive activation/engine_trace comparison logic. |
| `pkg/common/contracts.py` | `validate_with_schema()` — the schema validation function. |
| `pkg/common/deterministic.py` | `sha256_prefixed()`, `canonical_json_bytes()`, `sha256_file()`. Note: `sha256_file()` returns `sha256:<hex>` format (same as `expected_digest` in artifacts). |
| `pkg/networkdet/config.py` | `parse_net_config()` reads `manifest["network"]` — will crash after removal if not updated. |
| `pkg/networkdet/__init__.py` | `create_net_stack()` calls `parse_net_config`. Called by runner and capture. |
| `demo/run_demo.py` | Calls `create_net_stack()` — will break after network removal. |

### How the schema file works

`schemas/manifest.v1.schema.json` is a single line of minified JSON Schema (draft 2020-12). To edit it:

```bash
# Pretty-print it
python3 -m json.tool schemas/manifest.v1.schema.json > /tmp/pretty.json

# Edit /tmp/pretty.json

# Minify it back (CI requires canonical JSON)
python3 -c "import json,sys; json.dump(json.load(open('/tmp/pretty.json')), sys.stdout, separators=(',',':'), sort_keys=True)" > schemas/manifest.v1.schema.json
```

**Critical:** The schema uses `additionalProperties: false` on most objects. This means:
- You CANNOT add a field to a fixture/manifest without first adding it to the schema's `properties` block.
- You CANNOT remove a field from the schema's `properties` while fixtures still contain it.
- Schema + fixtures + manifests must be updated **atomically in the same commit**.

The schema gate (`scripts/ci/schema_gate.sh`) validates:
1. All positive fixtures pass validation
2. All negative fixtures fail validation
3. All schema files are canonical JSON (sorted keys, no whitespace)

### Critical rule: every commit must be green

Each commit must pass `make ci-pr`. This means **every commit that changes a schema must also update all fixtures, manifests, and tests that reference the changed fields in the same commit**. You cannot defer fixture updates to a later commit.

---

## Task 1: Remove dead manifest sections (one big commit)

**Goal:** Delete all fields that are never enforced, never passed to vLLM, and never used by downstream tools. This is done as ONE commit because the fields are interconnected and the schema/fixtures/code must stay in sync.

**Branch:** `manifest-v2` from `manifest-endpoint`

### What to remove from the manifest schema

Top-level removals:
- `network` (entire section: security_mode, egress_reproducibility, mtu, mss, tso, gso, checksum_offload, queue_mapping, ring_sizes, thread_affinity, internal_batching)
- `deterministic_dispatcher` (enabled, algorithm, request_order_source, replay_log_required)
- Remove both from the top-level `required` array

`model` removals:
- `model.remote_code` (commit, uri, digest)
- `model.requested_revision`
- `model.resolved_revision`
- `model.required_files` (entire array)
- Schema conditional: `if trust_remote_code == true then remote_code is required` — delete
- Remove `resolved_revision`, `required_files` from `model.required`

`runtime` removals:
- `runtime.batch_cardinality` (target_batch_size, min_requests, max_requests)
- `runtime.batch_policy`
- `runtime.engine_trace` (enabled, events)
- `runtime.nix_pin` (flake_ref, flake_hash)
- `runtime.allow_non_reproducible_egress`
- Schema conditional: `if engine_trace.enabled == true then events minItems 1` — delete
- Schema conditional: `if network.security_mode == "tls"` — delete
- Remove all five from `runtime.required`

`hardware_profile` removals:
- `hardware_profile.topology` (mode, node_count, rack_count, collective_fabric)
- `hardware_profile.nic` (entire section: model, pci_id, firmware, link_speed_gbps, offloads)
- `hardware_profile.gpu.vendor`
- `hardware_profile.gpu.pci_ids`
- Schema conditional: `if topology.mode in [replicated, TP, PP] then deterministic_dispatcher required` — delete
- Schema conditional: `if topology.mode in [TP, PP] then artifact_inputs must contain collective_stack` — delete
- Remove `topology`, `nic` from `hardware_profile.required`
- Remove `vendor`, `pci_ids` from `gpu.required`

`comparison` removals:
- `comparison.activations`
- `comparison.network_egress`
- Remove both from `comparison.required` (leaving only `tokens`, `logits`)

`artifact_inputs` changes:
- Lower `minItems` from 10 to a reasonable number (e.g. 4 — weights, config, tokenizer, serving_stack)
- Remove `allOf` constraints that require specific artifact types (`network_stack_binary`, `pmd_driver`, `nic_link_config`, `batching_policy`) — these are being deleted
- Add optional `path` and `role` fields to the artifact item schema (needed for Task 2):
  ```json
  "path": {"minLength": 1, "type": "string"},
  "role": {"enum": ["weights_shard", "config", "tokenizer", "generation_config", "chat_template", "prompt_formatter"], "type": "string"}
  ```

### What to update in the lockfile schema

`schemas/lockfile.v1.schema.json` has `allOf[contains]` constraints requiring artifact types: `network_stack_binary`, `pmd_driver`, `nic_link_config`, `batching_policy`. Remove these constraints. Also lower `artifacts.minItems` from 10.

### What to update in the run_bundle schema

`schemas/run_bundle.v1.schema.json` requires:
- `observables.activations` — remove from properties and required
- `observables.engine_trace` — remove from properties and required
- `observables.network_egress` — remove from properties and required
- `execution_trace_metadata.actual_batch_sizes` — remove
- `network_provenance` — remove from top-level required
- Any `batch_policy` references

### What to update in the verify_report schema

`schemas/verify_report.v1.schema.json` requires:
- Remove `"non_conformant_network"` from the `status` enum
- Remove `network_trace_diffs` property (first_diverging_frame, byte_offset_summaries) — dead after network_egress removal
- Remove `batch_trace_diffs` property — dead after engine_trace removal
- Remove the `allOf` conditional requiring these fields when status is `mismatch_outputs`
- Update any `required` arrays that reference these removed properties

### Code changes

**`cmd/runner/main.py`** — the biggest change:
- Remove `create_net_stack()` call and all network frame generation
- Remove `target_batch = manifest["runtime"]["batch_cardinality"]["target_batch_size"]` (lines 306, 446)
- Remove synthetic engine event generation (lines 320-331)
- Remove `actual_batch_sizes`, `batch_policy` from bundle metadata (lines 530-532)
- Remove `engine_trace` from bundle output (line 569)
- Remove `security_mode` from bundle metadata (line 554)
- Remove `_activations()` function and `act = _activations(toks)` call (line 316)
- Remove `"activations": act` from request_outputs (line 318)
- Remove NIC conformance check (lines 431-435)
- Remove `manifest["hardware_profile"]["topology"]["mode"]` reference (line 329)
- Remove `manifest["deterministic_dispatcher"]` references
- Make `create_net_stack` call conditional or remove it

**`cmd/runner/vllm_runner.py`**:
- Remove `target_batch` (line 116)
- Remove synthetic engine events (lines 144-173)
- Remove `activations` generation (line 115) and from request_outputs
- Remove `manifest["hardware_profile"]["topology"]["mode"]` reference (line 167)

**`cmd/capture/main.py`**:
- Remove `target_batch` (line 140)
- Remove `batch_size` in events (line 168)
- Remove `actual_batch_sizes`, `batch_policy` from bundle (lines 269-271)
- Remove engine_trace from bundle (line 307)
- Remove network references

**`cmd/resolver/main.py`**:
- Remove `manifest["model"]["remote_code"]` writing (lines 118-121)
- Remove `manifest["model"]["required_files"]` writing (line 117)
- Update `runtime_seed` dict (line 158) — remove `"network": manifest["network"]`
- Remove `_require_topology_artifacts()` function and its calls (lines 39-51, 138, 149)

**`pkg/common/hf_resolution.py`**:
- Remove `remote_code` from `ResolvedHF` dataclass (line 52)
- Remove `required_files` from `ResolvedHF` dataclass (line 50)
- Keep `_remote_code_digest()` function (used for artifact digest)
- Remove `required_files` list construction from `resolve_hf_model`
- The `add_file` function (line 538): stop building `required_files` entries, add `path` and `role` to artifact entries instead

**`cmd/verifier/main.py`**:
- Remove activation comparison logic
- Remove engine_trace comparison logic
- Remove network_egress comparison logic
- This is a substantial rewrite — the verifier is structured around 5 observables and you're removing 3. Carefully audit lines 140-214.

**`cmd/server/main.py`**:
- `_enforce_hardware()` (line 123): remove topology validation (lines 150-153), remove vendor check. Keep GPU model and count.
- `_handle_get_manifest()` GET response (line 602): remove `resolved_revision` reference — change to `weights_revision`
- Remove `_enforce_model_revision` reading `resolved_revision` — change to `weights_revision`

**`cmd/coordinator/main.py`**:
- Remove `manifest["deterministic_dispatcher"]["algorithm"]` (line 101). Take `--algorithm` as CLI arg instead.

**`cmd/runner/dispatcher.py`**:
- Remove `manifest["hardware_profile"]["topology"]["rack_count"]` (line 23). Take `--rack-count` as CLI arg.
- Remove `manifest["deterministic_dispatcher"]["algorithm"]` (line 24). Take `--algorithm` as CLI arg.

**`pkg/networkdet/config.py`**:
- Make `parse_net_config()` handle missing `network` section gracefully (return defaults or raise informative error).

**`demo/run_demo.py`**:
- Remove or guard `create_net_stack()` call.

### Fixture updates (same commit!)

**`tests/fixtures/positive/manifest.v1.example.json`** — remove all deleted fields. This is the most important fixture.

**`tests/fixtures/positive/lockfile.v1.example.json`** — remove network/nic/batching_policy artifact entries if they'd violate the updated lockfile schema.

**`tests/fixtures/positive/run_bundle.v1.example.json`** — remove activations, engine_trace, network_egress, actual_batch_sizes, network_provenance, batch_policy.

**`tests/fixtures/negative/`** — delete:
- `manifest.v1__remote_code_required_when_trusted.invalid.json`
- `manifest.v1__tls_requires_disabled_egress.invalid.json`
- `verify_report.v1__mismatch_missing_network_diffs.invalid.json` (if it depends on network_egress)

**`manifests/qwen3-1.7b.manifest.json`** — remove all deleted fields, collapse revisions.

### Test updates (same commit!)

**Delete these test files** (they test removed functionality):
- `tests/determinism/test_d5_network_egress.py`
- `tests/determinism/test_network_determinism.py`
- `tests/determinism/test_packet_determinism.py`
- `tests/determinism/test_multi_rack_topology.py`
- `tests/determinism/test_d4_tp_pp_trace.py`
- `tests/unit/test_topology.py`
- `tests/integration/test_resolver_topology_requirements.py`

**Update these test files** (they reference removed fields):
- `tests/determinism/test_d0_manifest_to_lockfile.py` — resolver reads `manifest["network"]`
- `tests/determinism/test_d1_builder_runtime_digest.py` — runtime_seed includes network
- `tests/determinism/test_d2_single_node_runner.py` — runs the synthetic runner
- `tests/determinism/test_d3_replicated_dispatch.py` — uses deterministic_dispatcher
- `tests/determinism/test_large_batch_determinism.py` — may reference batch_cardinality
- `tests/determinism/test_long_run_determinism.py` — bundle pipeline
- `tests/determinism/test_egress_bundle_pipeline.py` — network egress
- `tests/integration/test_builder_closure_profile.py` — may reference topology
- `tests/integration/test_resolver_hf_resolution.py` — tests `required_files`, `remote_code`, `requested_revision`, `resolved_revision`. Lines 186-187, 409-410, 483, 543, 567, 601, 660 all need updating.
- `tests/integration/test_runner_context_provenance.py` — bundle structure
- `tests/integration/test_runner_hardware_conformance.py` — hardware checks
- `tests/unit/test_manifest_endpoint.py` — `_enforce_model_revision` reads `resolved_revision` (line 83), `test_valid_manifest_passes` loads manifest
- `tests/unit/test_repo_layout.py` — may check for topology.py existence
- `tests/unit/test_spec_review_bugs.py` — may reference removed fields
- `tests/unit/test_networkdet_*.py` — these test the network stack itself, not the manifest fields. They may still pass if the network stack code isn't deleted, just disconnected from the manifest. Check each one.
- `tests/e2e/test_verifier_outputs.py` — runs the full pipeline (resolver → builder → runner → verifier) using the fixture manifest. The runner will stop producing activations/engine_trace/network_egress, which will crash the verifier. **Must update.**
- `tests/e2e/test_server_lifecycle.py` — loads the real manifest and runs the server. Will break if manifest is updated but test expectations aren't.
- `tests/e2e/test_manifest_endpoint_live.py` — loads the real manifest for POST /manifest testing.
- `tests/unit/test_warden_service.py` and `tests/integration/test_warden_inline.py` — may reference manifest fields being removed. Audit.
- `tests/chaos/test_chaos_scaffold.py` — needs auditing for removed field references.

**Also in the verifier code (`cmd/verifier/main.py`):**
- The `_first_network_byte_diff` helper function (lines 63-73) becomes dead code — delete it.
- The bundle-building section of `cmd/runner/main.py` at lines 462-467 writes `activations_path` and `activations_digest` — delete those lines too, along with the observable entry at lines 565-568.

### How to approach this

1. Pretty-print all **4** schemas to temp files (manifest, lockfile, run_bundle, verify_report)
2. Make all schema edits
3. Minify them back
4. Update all fixtures (positive AND negative, for ALL schemas)
5. Update the real manifest
6. Update all code files (grep for every removed field name across the entire codebase)
7. Update/delete tests
8. Run `make ci-pr` — fix anything that breaks
9. Commit everything together

**Commit:** `refactor: remove dead manifest fields (network, topology, dispatcher, batch, activations, nix_pin, remote_code)`

---

## Task 2: Add missing serving_engine fields and pass them to vLLM

**Depends on:** Task 1

### 2a. Add new fields to schema

Add these optional properties to `runtime.serving_engine` (remember: `additionalProperties: false` means you must add them to the `properties` block):

```json
"quantization": {"type": ["string", "null"], "enum": ["awq", "gptq", "bitsandbytes", "fp8", null]},
"load_format": {"type": "string", "enum": ["auto", "safetensors", "pt", "gguf"]},
"kv_cache_dtype": {"type": "string", "enum": ["auto", "fp8", "int8"]},
"max_num_batched_tokens": {"type": ["integer", "null"], "minimum": 1},
"block_size": {"type": ["integer", "null"], "minimum": 1},
"enable_prefix_caching": {"type": "boolean"},
"enable_chunked_prefill": {"type": "boolean"},
"scheduling_policy": {"type": "string", "enum": ["fcfs", "priority"]},
"disable_sliding_window": {"type": "boolean"},
"tensor_parallel_size": {"type": "integer", "minimum": 1},
"pipeline_parallel_size": {"type": "integer", "minimum": 1},
"disable_custom_all_reduce": {"type": "boolean"}
```

Do NOT add to `required`. Also add `cublas_workspace_config` (string) and `pythonhashseed` (string) to `runtime.deterministic_knobs`.

### 2b. Pass them to vLLM

In `cmd/server/main.py` `_build_vllm_cmd()`, add:

```python
quantization = engine.get("quantization")
if quantization:
    cmd.extend(["--quantization", quantization])

load_format = engine.get("load_format")
if load_format:
    cmd.extend(["--load-format", load_format])

kv_cache_dtype = engine.get("kv_cache_dtype")
if kv_cache_dtype:
    cmd.extend(["--kv-cache-dtype", kv_cache_dtype])

max_num_batched_tokens = engine.get("max_num_batched_tokens")
if max_num_batched_tokens:
    cmd.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])

block_size = engine.get("block_size")
if block_size:
    cmd.extend(["--block-size", str(block_size)])

if engine.get("enable_prefix_caching"):
    cmd.append("--enable-prefix-caching")

if engine.get("enable_chunked_prefill"):
    cmd.append("--enable-chunked-prefill")

scheduling_policy = engine.get("scheduling_policy")
if scheduling_policy:
    cmd.extend(["--scheduling-policy", scheduling_policy])

if engine.get("disable_sliding_window"):
    cmd.append("--disable-sliding-window")

tp = engine.get("tensor_parallel_size")
if tp and tp > 1:
    cmd.extend(["--tensor-parallel-size", str(tp)])

pp = engine.get("pipeline_parallel_size")
if pp and pp > 1:
    cmd.extend(["--pipeline-parallel-size", str(pp)])

if engine.get("disable_custom_all_reduce"):
    cmd.append("--disable-custom-all-reduce")
```

In `_set_deterministic_env()`:
```python
os.environ["CUBLAS_WORKSPACE_CONFIG"] = knobs.get("cublas_workspace_config", ":4096:8")
os.environ["PYTHONHASHSEED"] = str(knobs.get("pythonhashseed", "0"))
```

### 2c. Tests

Write unit tests in `tests/unit/test_manifest_endpoint.py`:

```python
class TestBuildVllmCmd(unittest.TestCase):
    def test_quantization_flag_present(self):
        m = _load_manifest()
        m["runtime"]["serving_engine"]["quantization"] = "awq"
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--quantization", cmd)
        self.assertEqual(cmd[cmd.index("--quantization") + 1], "awq")

    def test_quantization_flag_absent(self):
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--quantization", cmd)

    # Write similar pairs for each new flag
```

**Import note:** Python's built-in `cmd` module conflicts with our `cmd/` directory. Import server functions like this:
```python
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "server_main", REPO_ROOT / "cmd" / "server" / "main.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_build_vllm_cmd = _mod._build_vllm_cmd
```

**Commit:** `feat: add serving_engine fields and pass to vLLM CLI`

---

## Task 3: Add container image digest verification

**Depends on:** Task 1

### 3a. Add field to schema

Add to `runtime` (optional):
```json
"container_image_digest": {"type": "string", "pattern": "^sha256:[a-f0-9]{64}$"}
```

### 3b. Bake digest into container

The OCI image digest is a chicken-and-egg problem (digest isn't known until the image is built). Two options:

**Option A (recommended):** After building the OCI image with `nix build .#oci`, run a post-build step that reads the digest and writes it to a manifest. The server reads the manifest at boot.

**Option B:** Bake a build-time identifier into the container as an env var in the flake (e.g. `CONTAINER_BUILD_HASH` from the nix closure hash), and verify that instead. Not the image digest but still pins the closure.

### 3c. Verify at startup

In `_start_vllm()`:
```python
expected_digest = manifest.get("runtime", {}).get("container_image_digest")
if expected_digest:
    actual_digest = os.environ.get("CONTAINER_IMAGE_DIGEST", "")
    if actual_digest and actual_digest != expected_digest:
        raise ValidationError(
            f"Container image digest mismatch: expected {expected_digest}, got {actual_digest}"
        )
```

**Commit:** `feat: add container_image_digest verification`

---

## Task 4: Add GPU driver version verification

**Depends on:** Task 1

In `cmd/server/main.py` `_enforce_hardware()`, add:

```python
expected_driver = gpu.get("driver_version")
if expected_driver:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        actual_driver = result.stdout.strip().splitlines()[0].strip()
        if actual_driver != expected_driver:
            warnings.append(f"GPU driver mismatch: manifest={expected_driver}, actual={actual_driver}")
    except Exception:
        warnings.append("Could not query GPU driver version")

expected_cuda = gpu.get("cuda_driver_version")
if expected_cuda:
    import torch
    actual_cuda = torch.version.cuda or "unknown"
    if actual_cuda != expected_cuda:
        warnings.append(f"CUDA version mismatch: manifest={expected_cuda}, actual={actual_cuda}")
```

**Test:** Mock `subprocess.run` to return a known driver version.

**Commit:** `feat: verify GPU driver and CUDA versions against manifest`

---

## Task 5: Local file hash verification

**Depends on:** Task 1 (artifact_inputs has `path` and `role` fields)

### 5a. Understand the HF cache layout

HuggingFace caches files in:
```
~/.cache/huggingface/hub/models--Org--Model/snapshots/<commit_sha>/
```

**Important:** The snapshot directory contains **symlinks** to blob files:
```
snapshots/<commit>/config.json -> ../../blobs/<sha256>
```

`Path.is_file()` follows symlinks, so this works transparently. `sha256_file()` also follows symlinks and hashes the blob content.

**Edge case:** If a download was interrupted, the symlink may point to a partial blob. Check `size_bytes` from the artifact against `file_path.stat().st_size` as a quick sanity check.

**Edge case:** If `RUNNER_MODEL_PATH` env var is set (line 195 in main.py), vLLM loads from that path instead of HF cache. The hash verification function should check this path first.

### 5b. Implement verification

```python
from pkg.common.deterministic import sha256_file

def _verify_model_artifacts(manifest: dict, report: dict) -> None:
    model = manifest["model"]
    revision = model.get("weights_revision")
    if not revision:
        report["warnings"].append("No weights_revision — skipping file verification")
        return

    # Check RUNNER_MODEL_PATH first, then HF cache
    model_path = os.environ.get("RUNNER_MODEL_PATH")
    if model_path and Path(model_path).is_dir():
        cache_path = Path(model_path)
    else:
        repo_id = model["source"].removeprefix("hf://")
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        cache_path = cache_dir / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision
        if not cache_path.is_dir():
            report["warnings"].append(f"HF cache not found for {repo_id}@{revision[:12]}")
            return

    model_types = {"model_weights", "model_config", "tokenizer", "generation_config", "chat_template"}
    for artifact in manifest.get("artifact_inputs", []):
        if artifact.get("artifact_type") not in model_types:
            continue
        expected = artifact.get("expected_digest")
        path = artifact.get("path")
        if not expected or not path:
            continue

        file_path = cache_path / path
        if not file_path.is_file():
            report["warnings"].append(f"File not found in cache: {path}")
            continue

        # Quick size check
        expected_size = artifact.get("size_bytes")
        actual_size = file_path.stat().st_size
        if expected_size and actual_size != expected_size:
            raise ValidationError(
                f"File size mismatch for {path}: expected {expected_size}, got {actual_size} "
                f"(possible incomplete download)"
            )

        # Full hash check — sha256_file returns "sha256:<hex>"
        actual = sha256_file(file_path)
        if actual != expected:
            raise ValidationError(
                f"File digest mismatch for {path}: expected {expected[:24]}..., got {actual[:24]}..."
            )
        report["enforced"].append(f"verified {path}")
```

Call from `_start_vllm()` after hardware checks, before launching vLLM.

### 5c. Test

Create a temp dir with fake files, construct artifact_inputs with correct/wrong digests:
```python
def test_correct_digest_passes(self):
    # write a file, compute its sha256, set expected_digest to match
    ...

def test_wrong_digest_raises(self):
    # write a file, set expected_digest to a different hash
    with self.assertRaises(ValidationError):
        ...

def test_missing_file_warns(self):
    # artifact points to nonexistent file
    ...

def test_missing_revision_skips(self):
    # no weights_revision in manifest
    ...
```

**Commit:** `feat: verify model file digests from artifact_inputs before serving`

---

## Task 6: Final fixture and manifest update

**Depends on:** All previous tasks

Update `manifests/qwen3-1.7b.manifest.json` and all fixtures to include any new fields from Tasks 2-5 (serving_engine fields, cublas_workspace_config, etc.).

Add new negative fixtures:
- `manifest.v1__bad_quantization.invalid.json` — quantization set to invalid value

Run `make ci-pr`. All gates must pass.

**Commit:** `chore: update manifests and fixtures for v2 schema`

---

## Testing principles

1. **Schema + fixtures + code in the same commit.** Never change a schema without updating the fixtures and manifests that reference it. The CI will catch you.

2. **One assertion per test method.** Each test should fail for exactly one reason.

3. **Test the boundaries.** For enum fields: valid value, invalid value, absent. For integers with minimums: 0, 1, valid.

4. **Schema tests are cheap.** `validate_with_schema` takes <1ms. Write lots of them.

5. **Don't mock what you can construct.** Call `_build_vllm_cmd` with a real manifest dict and check the list.

6. **Import server functions carefully.** Python's built-in `cmd` module conflicts. Use `importlib.util.spec_from_file_location`.

7. **After every code edit, run tests immediately.** Don't batch up changes — fix as you go.

---

## Commit sequence

```
1.  refactor: remove dead manifest fields (network, topology, dispatcher, batch, activations, nix_pin, remote_code)
2.  feat: add serving_engine fields and pass to vLLM CLI
3.  feat: add container_image_digest verification
4.  feat: verify GPU driver and CUDA versions against manifest
5.  feat: verify model file digests from artifact_inputs before serving
6.  chore: update manifests and fixtures for v2 schema
```

Task 1 is the hardest (touches the most files). Tasks 2-5 are independent of each other and can be done in any order. Task 6 is cleanup.
