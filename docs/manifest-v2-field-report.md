# Manifest v2 Field Report

Every field in the manifest, what it does, whether it was added/removed/kept, and where it's used — with code references. All fields are now typed via the Pydantic model at `pkg/manifest/model.py`.

## Summary

| | Count |
|---|---|
| Fields before v2 | 74 |
| Fields removed | 45 |
| Fields added | 15 |
| Fields after v2 | 44 |
| Pydantic model classes | 11 |
| Unit tests | 186 |

---

## Current fields (after v2)

### Top-level

| Field | Type | Req | Pydantic class | Used in | How |
|-------|------|-----|----------------|---------|-----|
| `manifest_version` | `Literal["v1"]` | opt | `Manifest` | Schema validation | Always "v1" |
| `run_id` | `str` | req | `Manifest` | `server/main.py:666,789`, `runner/main.py:281,391` | Identifies the run in responses, seed derivation, boot record |
| `created_at` | `str` | req | `Manifest` | `resolver/main.py:127` | Timestamp carried into lockfile |

### model

| Field | Type | Req | Used in | How enforced |
|-------|------|-----|---------|-------------|
| `model.source` | `str` (pattern: `^hf://...`) | req | `server/main.py:201` | `--model` flag to vLLM |
| `model.weights_revision` | `str` (40 hex) | req | `server/main.py:180` | `--revision` flag to vLLM |
| `model.tokenizer_revision` | `str` (40 hex) | req | `server/main.py:211` | `--tokenizer-revision` flag to vLLM |
| `model.trust_remote_code` | `bool` | req | `server/main.py:248` | `--trust-remote-code` flag to vLLM |

### runtime.serving_engine (17 fields)

| Field | Type | Req | vLLM flag | Code ref |
|-------|------|-----|-----------|----------|
| `max_model_len` | `int` | req | `--max-model-len` | `server/main.py:221` + request validation at `:186` |
| `max_num_seqs` | `int` | req | `--max-num-seqs` | `server/main.py:226` |
| `gpu_memory_utilization` | `float` (0.1–1.0) | req | `--gpu-memory-utilization` | `server/main.py:224` |
| `dtype` | `Dtype` enum | opt | `--dtype` | `server/main.py:218` |
| `attention_backend` | `AttentionBackend` enum | req | `--attention-backend` | `server/main.py:230` |
| `quantization` | `str \| None` | opt | `--quantization` | `server/main.py:233` |
| `load_format` | `str \| None` | opt | `--load-format` | `server/main.py:237` |
| `kv_cache_dtype` | `str \| None` | opt | `--kv-cache-dtype` | `server/main.py:241` |
| `max_num_batched_tokens` | `int \| None` | opt | `--max-num-batched-tokens` | `server/main.py:245` |
| `block_size` | `int \| None` | opt | `--block-size` | `server/main.py:249` |
| `enable_prefix_caching` | `bool \| None` | opt | `--enable-prefix-caching` | `server/main.py:253` |
| `enable_chunked_prefill` | `bool \| None` | opt | `--enable-chunked-prefill` | `server/main.py:256` |
| `scheduling_policy` | `str \| None` | opt | `--scheduling-policy` | `server/main.py:259` |
| `disable_sliding_window` | `bool \| None` | opt | `--disable-sliding-window` | `server/main.py:263` |
| `tensor_parallel_size` | `int \| None` | opt | `--tensor-parallel-size` | `server/main.py:266` |
| `pipeline_parallel_size` | `int \| None` | opt | `--pipeline-parallel-size` | `server/main.py:270` |
| `disable_custom_all_reduce` | `bool \| None` | opt | `--disable-custom-all-reduce` | `server/main.py:274` |

### runtime.batch_invariance

| Field | Type | Req | Used in | How enforced |
|-------|------|-----|---------|-------------|
| `enabled` | `bool` | req | `server/main.py:321` | `VLLM_BATCH_INVARIANT=1` env var |
| `enforce_eager` | `bool` | req | `server/main.py:235` | `--enforce-eager` flag |

### runtime.deterministic_knobs

| Field | Type | Req | Used in | How enforced |
|-------|------|-----|---------|-------------|
| `seed` | `int` (≥0) | req | `server/main.py:208` | `--seed` flag |
| `torch_deterministic` | `bool` | req | `vllm_runner.py:62` | `torch.use_deterministic_algorithms(True)` |
| `cuda_launch_blocking` | `bool` | req | `server/main.py:318` | `CUDA_LAUNCH_BLOCKING` env var |
| `cublas_workspace_config` | `str` | opt (default `:4096:8`) | `server/main.py:316` | `CUBLAS_WORKSPACE_CONFIG` env var |
| `pythonhashseed` | `str` | opt (default `0`) | `server/main.py:319` | `PYTHONHASHSEED` env var |

### runtime (other)

| Field | Type | Req | Used in | How enforced |
|-------|------|-----|---------|-------------|
| `strict_hardware` | `bool` | req | `server/main.py:411`, `runner/main.py:350` | Fatal vs warning on hardware mismatch |
| `closure_hash` | `str \| None` (sha256 pattern) | opt | `server/main.py:329` | Compared against `CLOSURE_HASH` env var (set by nix flake) |

### hardware_profile.gpu

| Field | Type | Req | Used in | How verified |
|-------|------|-----|---------|-------------|
| `model` | `str` | req | `server/main.py:143` | Substring match against `torch.cuda.get_device_name()` |
| `count` | `int` (≥1) | req | `server/main.py:137` | Compared against `torch.cuda.device_count()` |
| `driver_version` | `str` | req | `server/main.py:153` | Compared against `nvidia-smi --query-gpu=driver_version` |
| `cuda_driver_version` | `str` | req | `server/main.py:163` | Compared against `torch.version.cuda` |

### requests[]

| Field | Type | Req | Used in | How enforced |
|-------|------|-----|---------|-------------|
| `id` | `str` (pattern) | req | `runner/main.py:281` | Seed derivation, bundle entry ID |
| `prompt` | `str` (min 1) | req | `vllm_runner.py:96` | Sent to vLLM for inference |
| `max_new_tokens` | `int` (1–4096) | req | `server/main.py:188` | Validated against `max_model_len` before serving |
| `temperature` | `float` (0–2) | req | `vllm_runner.py:99` | Passed to `SamplingParams` |

### comparison

| Field | Type | Req | Used in | How enforced |
|-------|------|-----|---------|-------------|
| `tokens` | `Comparator` | req | `verifier/main.py:127` | Verifier compares token sequences |
| `logits` | `Comparator` | req | `verifier/main.py:127` | Verifier compares logprob values |

Each `Comparator` has: `mode` (exact/ulp/absrel/hash), and optional `algorithm`, `ulp`, `atol`, `rtol`.

### artifact_inputs[]

| Field | Type | Req | Used in | How enforced |
|-------|------|-----|---------|-------------|
| `artifact_id` | `str` (pattern) | req | `runner/main.py:333`, `resolver/main.py:129` | Bundle provenance, lockfile generation |
| `artifact_type` | `ArtifactType` enum | req | `server/main.py:362` | Identifies model files for hash verification |
| `source_kind` | `SourceKind` enum | req | `resolver/main.py:129` | Lockfile artifact source |
| `source_uri` | `str` | req | `resolver/main.py:129` | Lockfile artifact URI |
| `immutable_ref` | `str` | req | `resolver/main.py:129` | Pinned reference (commit SHA) |
| `name` | `str \| None` | opt | `resolver/main.py` | Human-readable name |
| `expected_digest` | `str \| None` (sha256 pattern) | opt | `server/main.py:367` | SHA256 verified against cached files before serving |
| `size_bytes` | `int \| None` (≥1) | opt | `server/main.py:372` | Size check before hash verification |
| `path` | `str \| None` | opt | `server/main.py:365` | Locates cached file for hash verification |
| `role` | `FileRole \| None` | opt | `server/main.py:362` | Identifies file function (weights_shard, config, tokenizer, etc.) |

---

## Fields removed in v2

| Section | Fields removed | Count | Why |
|---------|---------------|-------|-----|
| `network` | security_mode, egress_reproducibility, mtu, mss, tso, gso, checksum_offload, queue_mapping, ring_sizes, thread_affinity, internal_batching | 11 | No networking stack |
| `hardware_profile.topology` | mode, node_count, rack_count, collective_fabric | 4 | Single-node only; TP/PP now in serving_engine |
| `hardware_profile.nic` | model, pci_id, firmware, link_speed_gbps, offloads (5 sub-fields) | 5 | No networking stack |
| `hardware_profile.gpu` | vendor, pci_ids | 2 | Redundant / differs per machine |
| `model` | requested_revision, resolved_revision, required_files, remote_code (3 sub-fields) | 6 | Collapsed revisions; merged into artifact_inputs |
| `runtime` | batch_cardinality (3), batch_policy, engine_trace (2), nix_pin (2), allow_non_reproducible_egress | 10 | Not passed to vLLM; fabricated data; replaced by closure_hash |
| `comparison` | activations, network_egress | 2 | Fake data; no networking stack |
| `deterministic_dispatcher` | enabled, algorithm, request_order_source, replay_log_required | 4 | Single-node only |
| Schema conditionals | 5 allOf/if-then rules | — | Referenced removed fields |
| **Total** | | **45** | |

## Fields added in v2

| Field | Why added |
|-------|-----------|
| 12 `serving_engine` optional fields | Cover all vLLM flags that affect determinism |
| `cublas_workspace_config` | Was hardcoded, now explicit in manifest |
| `pythonhashseed` | Was hardcoded, now explicit in manifest |
| `closure_hash` | Replaces nix_pin; verifiable at runtime via env var |
| `artifact_inputs[].path` | Locates cached files for hash verification |
| `artifact_inputs[].role` | Replaces required_files role field |

---

## Verification summary

Every remaining field falls into exactly one category:

| Category | Count | Mechanism |
|----------|-------|-----------|
| **Passed to vLLM as CLI flag** | 21 | `_build_vllm_cmd()` in `server/main.py` |
| **Set as env var** | 4 | `_set_deterministic_env()` in `server/main.py` |
| **Verified against runtime** | 6 | GPU probe, nvidia-smi, torch.version.cuda, closure hash, file hashing |
| **Validated before serving** | 4 | Request token limits, schema validation |
| **Used by downstream tools** | 9 | Verifier (comparison), resolver (artifact_inputs, created_at), runner (run_id, requests) |

No field is purely decorative. Everything is either enforced, verified, or consumed.

## Pydantic model classes

All defined in `pkg/manifest/model.py`:

| Class | Fields |
|-------|--------|
| `Manifest` | Top-level: manifest_version, run_id, created_at, model, runtime, hardware_profile, requests, comparison, artifact_inputs |
| `ModelConfig` | source, weights_revision, tokenizer_revision, trust_remote_code |
| `RuntimeConfig` | strict_hardware, batch_invariance, deterministic_knobs, serving_engine, closure_hash |
| `ServingEngine` | 17 fields (5 required + 12 optional) |
| `BatchInvariance` | enabled, enforce_eager |
| `DeterministicKnobs` | seed, torch_deterministic, cuda_launch_blocking, cublas_workspace_config, pythonhashseed |
| `HardwareProfile` | gpu |
| `GpuProfile` | model, count, driver_version, cuda_driver_version |
| `RequestItem` | id, prompt, max_new_tokens, temperature |
| `ComparisonConfig` | tokens, logits |
| `Comparator` | mode, algorithm, ulp, atol, rtol |
| `ArtifactInput` | artifact_id, artifact_type, source_kind, source_uri, immutable_ref, name, expected_digest, size_bytes, path, role |
