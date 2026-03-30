# Manifest Field Registry

Every field in `manifest.v1.schema.json` is listed below with its **effect**
(what the code does with it), **verification** (how correctness is checked),
and **production use** (where it matters at runtime).

Status labels:
- **validated** — rejected if malformed (schema + Pydantic)
- **passed-to-vllm** — appears in vLLM CLI args or engine kwargs
- **env-var** — set as an environment variable
- **controls-branch** — gates a code path with observable effect
- **recorded** — written into run bundle / lockfile / report
- **compared** — checked against actual runtime state
- **used-in-computation** — feeds into a deterministic algorithm
- **DEAD** — parsed but never read by any production code

---

## Top-Level

### `manifest_version`
- **Effect:** validated
- **Verification:** Pydantic `Literal["v1"]` rejects anything other than `"v1"`. JSON Schema `const` enforces the same.
- **Production use:** None beyond admission control. Not read after parsing.
- **Test:** `test_manifest_model::TestManifestRejectsInvalid::test_rejects_bad_manifest_version`

### `run_id`
- **Effect:** validated, used-in-computation, recorded
- **Verification:** Pydantic pattern `^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$`. Missing `run_id` rejected.
- **Production use:** Seeds per-request RNG in runner (`cmd/runner/main.py:281`). Recorded in run bundle, verify report, and capture session. Returned in `/manifest` and `/run` responses.
- **Test:** `test_manifest_model::TestManifestRejectsInvalid::test_rejects_missing_run_id`

### `created_at`
- **Effect:** validated, used-in-computation, recorded
- **Verification:** Pydantic `str` (schema `format: date-time`). No runtime format enforcement beyond schema.
- **Production use:** Used as deterministic timestamp for lockfile generation in resolver (`cmd/resolver/main.py:131`). Displayed in bundle visualizer.
- **Test:** Only in coverage manifests. No dedicated behavioral test.
- **Gap:** No test verifies the resolver actually uses this timestamp.

---

## `model`

### `model.source`
- **Effect:** validated, passed-to-vllm, used-in-computation
- **Verification:** Pydantic pattern `^hf://[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$`.
- **Production use:** `hf://` prefix stripped, remainder passed as `--model` to vLLM (`cmd/server/main.py:168,172`). Used in resolver for HF model resolution (`cmd/resolver/main.py:85`). Used to locate HF cache directory for artifact verification.
- **Test:** `test_manifest_model` (pattern), `test_manifest_endpoint::TestSchemaValidation::test_bad_model_source_rejected`

### `model.weights_revision`
- **Effect:** validated, passed-to-vllm, compared
- **Verification:** Pydantic pattern `^[a-f0-9]{40}$` (40-char hex SHA).
- **Production use:** Passed as `--revision` to vLLM (`cmd/server/main.py:182`). Used to locate snapshot directory in HF cache for artifact digest verification (`cmd/server/main.py:321`). Overwritten by resolver with resolved revision (`cmd/resolver/main.py:100`).
- **Test:** `test_manifest_model::test_rejects_bad_weights_revision`, `test_manifest_endpoint::TestVerifyModelArtifacts`

### `model.tokenizer_revision`
- **Effect:** validated, passed-to-vllm
- **Verification:** Pydantic pattern `^[a-f0-9]{40}$`.
- **Production use:** Passed as `--tokenizer-revision` to vLLM, falls back to `weights_revision` if falsy (`cmd/server/main.py:183`). Overwritten by resolver (`cmd/resolver/main.py:101`).
- **Test:** Coverage manifests only. No dedicated behavioral test.
- **Gap:** No test verifies it appears in the vLLM command or that fallback to `weights_revision` works.

### `model.trust_remote_code`
- **Effect:** validated, passed-to-vllm, controls-branch
- **Verification:** Pydantic `bool`.
- **Production use:** When true, adds `--trust-remote-code` to vLLM CLI (`cmd/server/main.py:241`). Passed to resolver for remote code hashing (`cmd/resolver/main.py:92`). Passed to vLLM LLM constructor in runner (`cmd/runner/vllm_runner.py:75`).
- **Test:** `test_manifest_model` (parse). No test for the vLLM flag being added/omitted.
- **Gap:** No behavioral test verifying the flag appears in vLLM command.

---

## `runtime`

### `runtime.strict_hardware`
- **Effect:** validated, controls-branch
- **Verification:** Pydantic `bool`.
- **Production use:** If true and hardware mismatches, raises `ValidationError` refusing to start (`cmd/server/main.py:378`). If false, logs warnings only.
- **Test:** `test_manifest_endpoint::TestCheckHardware` (pure function tests), `test_runner_hardware_conformance` (integration, strict refuse + non-strict label)

### `runtime.closure_hash`
- **Effect:** validated, compared
- **Verification:** Pydantic pattern `^sha256:[a-f0-9]{64}$`. Optional field.
- **Production use:** Compared against `CLOSURE_HASH` env var. Mismatch raises `ValidationError` (`cmd/server/main.py:292-304`).
- **Test:** `test_manifest_endpoint::TestClosureHash` (match, mismatch, absent, env var missing)

---

## `runtime.batch_invariance`

### `runtime.batch_invariance.enabled`
- **Effect:** validated, env-var, controls-branch
- **Verification:** Pydantic `bool`.
- **Production use:** Sets `VLLM_BATCH_INVARIANT=1` env var (`cmd/server/main.py:286-289`). Passed as `enable_batch_invariance=True` to vLLM LLM constructor (`cmd/runner/vllm_runner.py:87-88`). Recorded in enforcement report and summary.
- **Test:** Coverage manifests only. No behavioral test.
- **Gap:** No test verifies the env var is set or the LLM kwarg is passed.

### `runtime.batch_invariance.enforce_eager`
- **Effect:** validated, passed-to-vllm
- **Verification:** Pydantic `bool`.
- **Production use:** Adds `--enforce-eager` flag to vLLM CLI (`cmd/server/main.py:198-199`). Passed as `enforce_eager=True` to LLM constructor (`cmd/runner/vllm_runner.py:79-80`).
- **Test:** Coverage manifests only. No behavioral test.
- **Gap:** No test verifies the flag appears in vLLM command. (The existing `TestBuildVllmCmd` tests skip this field.)

---

## `runtime.deterministic_knobs`

### `runtime.deterministic_knobs.seed`
- **Effect:** validated, passed-to-vllm, used-in-computation
- **Verification:** Pydantic `int` with `ge=0`.
- **Production use:** Passed as `--seed` to vLLM (`cmd/server/main.py:175`). Used to derive per-request seeds in runner (`cmd/runner/main.py:281`). Passed to `SamplingParams` (`cmd/runner/vllm_runner.py:68`).
- **Test:** `test_manifest_model::test_rejects_negative_seed`

### `runtime.deterministic_knobs.torch_deterministic`
- **Effect:** validated, controls-branch
- **Verification:** Pydantic `bool`.
- **Production use:** Calls `torch.use_deterministic_algorithms(True)` and disables CUDA benchmarking in runner (`cmd/runner/vllm_runner.py:62-65`).
- **Test:** Coverage manifests only. No behavioral test.
- **Gap:** No test verifies `torch.use_deterministic_algorithms` is called.

### `runtime.deterministic_knobs.cuda_launch_blocking`
- **Effect:** validated, env-var
- **Verification:** Pydantic `bool`.
- **Production use:** Set as `CUDA_LAUNCH_BLOCKING` env var in server (`cmd/server/main.py:283`) and runner (`cmd/runner/main.py:447`, `cmd/runner/vllm_runner.py:19`).
- **Test:** Coverage manifests only. No behavioral test.
- **Gap:** No test verifies the env var is set.

### `runtime.deterministic_knobs.cublas_workspace_config`
- **Effect:** validated, env-var (server only)
- **Verification:** Pydantic `str | None`.
- **Production use:** Set as `CUBLAS_WORKSPACE_CONFIG` in server, defaults to `":4096:8"` (`cmd/server/main.py:282`).
- **Test:** `test_manifest_endpoint::TestSetDeterministicEnv` (explicit value + default)
- **Issue:** Runner and vllm_runner hardcode `":4096:8"` — manifest value is **ignored** outside the server path (`cmd/runner/main.py:446`, `cmd/runner/vllm_runner.py:18`).

### `runtime.deterministic_knobs.pythonhashseed`
- **Effect:** validated, env-var (server only)
- **Verification:** Pydantic `str | None`.
- **Production use:** Set as `PYTHONHASHSEED` in server, defaults to `"0"` (`cmd/server/main.py:284`).
- **Test:** `test_manifest_endpoint::TestSetDeterministicEnv` (explicit value + default)
- **Issue:** Runner and vllm_runner hardcode `"0"` — manifest value is **ignored** outside the server path (`cmd/runner/vllm_runner.py:20`).

---

## `runtime.serving_engine`

All serving engine fields follow the same pattern: validated by Pydantic, then mapped to a vLLM CLI flag or engine kwarg.

### `runtime.serving_engine.max_model_len`
- **Effect:** validated, passed-to-vllm, used-in-computation
- **Verification:** Pydantic `int` with `ge=1`.
- **Production use:** Passed as `--max-model-len` (`cmd/server/main.py:189`). Used to validate all `requests[].max_new_tokens` fit within context (`cmd/server/main.py:151-154`).
- **Test:** `test_manifest_endpoint::TestRequestValidation`

### `runtime.serving_engine.max_num_seqs`
- **Effect:** validated, passed-to-vllm
- **Verification:** Pydantic `int` with `ge=1`.
- **Production use:** Passed as `--max-num-seqs` (`cmd/server/main.py:192-193`).
- **Test:** No behavioral test. Only coverage manifests.
- **Gap:** No test verifies flag appears in command.

### `runtime.serving_engine.gpu_memory_utilization`
- **Effect:** validated, passed-to-vllm
- **Verification:** Pydantic `float` with `ge=0.1, le=1.0`.
- **Production use:** Passed as `--gpu-memory-utilization` (`cmd/server/main.py:190`).
- **Test:** No behavioral test. Only coverage manifests.
- **Gap:** No test verifies flag appears in command.

### `runtime.serving_engine.dtype`
- **Effect:** validated, passed-to-vllm
- **Verification:** Pydantic `Dtype` enum (auto, float16, bfloat16, float32).
- **Production use:** Passed as `--dtype` (`cmd/server/main.py:188`).
- **Test:** No behavioral test for the flag. Enum coverage in field coverage test.
- **Gap:** No test verifies flag appears in command.

### `runtime.serving_engine.attention_backend`
- **Effect:** validated, passed-to-vllm
- **Verification:** Pydantic `AttentionBackend` enum.
- **Production use:** Passed as `--attention-backend` (`cmd/server/main.py:195`).
- **Test:** No behavioral test for the flag. Enum coverage in field coverage test.
- **Gap:** No test verifies flag appears in command.

### `runtime.serving_engine.quantization`
- **Effect:** validated, passed-to-vllm
- **Verification:** Pydantic `Quantization | None` enum.
- **Production use:** When set, passed as `--quantization` (`cmd/server/main.py:202-203`).
- **Test:** `test_manifest_endpoint::TestBuildVllmCmd::test_quantization_present/absent`

### `runtime.serving_engine.load_format`
- **Effect:** validated, passed-to-vllm
- **Verification:** Pydantic `LoadFormat | None` enum.
- **Production use:** When set, passed as `--load-format` (`cmd/server/main.py:205-206`).
- **Test:** `test_manifest_endpoint::TestBuildVllmCmd::test_load_format_present/absent`

### `runtime.serving_engine.kv_cache_dtype`
- **Effect:** validated, passed-to-vllm
- **Verification:** Pydantic `KVCacheDtype | None` enum.
- **Production use:** When set, passed as `--kv-cache-dtype` (`cmd/server/main.py:208-209`).
- **Test:** `test_manifest_endpoint::TestBuildVllmCmd::test_kv_cache_dtype_present/absent`

### `runtime.serving_engine.max_num_batched_tokens`
- **Effect:** validated, passed-to-vllm
- **Verification:** Pydantic `int | None` with `ge=1`.
- **Production use:** When set, passed as `--max-num-batched-tokens` (`cmd/server/main.py:211-212`).
- **Test:** `test_manifest_endpoint::TestBuildVllmCmd::test_max_num_batched_tokens_present/absent`

### `runtime.serving_engine.block_size`
- **Effect:** validated, passed-to-vllm
- **Verification:** Pydantic `int | None` with `ge=1`.
- **Production use:** When set, passed as `--block-size` (`cmd/server/main.py:214-215`).
- **Test:** `test_manifest_endpoint::TestBuildVllmCmd::test_block_size_present/absent`

### `runtime.serving_engine.enable_prefix_caching`
- **Effect:** validated, passed-to-vllm
- **Verification:** Pydantic `bool | None`.
- **Production use:** When true, adds `--enable-prefix-caching` (`cmd/server/main.py:217-218`).
- **Test:** `test_manifest_endpoint::TestBuildVllmCmd::test_enable_prefix_caching_present/absent`

### `runtime.serving_engine.enable_chunked_prefill`
- **Effect:** validated, passed-to-vllm
- **Verification:** Pydantic `bool | None`.
- **Production use:** When true, adds `--enable-chunked-prefill` (`cmd/server/main.py:220-221`).
- **Test:** `test_manifest_endpoint::TestBuildVllmCmd::test_enable_chunked_prefill_present/absent`

### `runtime.serving_engine.scheduling_policy`
- **Effect:** validated, passed-to-vllm
- **Verification:** Pydantic `SchedulingPolicy | None` enum.
- **Production use:** When set, passed as `--scheduling-policy` (`cmd/server/main.py:223-224`).
- **Test:** `test_manifest_endpoint::TestBuildVllmCmd::test_scheduling_policy_present/absent`

### `runtime.serving_engine.disable_sliding_window`
- **Effect:** validated, passed-to-vllm
- **Verification:** Pydantic `bool | None`.
- **Production use:** When true, adds `--disable-sliding-window` (`cmd/server/main.py:226-227`).
- **Test:** `test_manifest_endpoint::TestBuildVllmCmd::test_disable_sliding_window_present/absent`

### `runtime.serving_engine.tensor_parallel_size`
- **Effect:** validated, passed-to-vllm
- **Verification:** Pydantic `int | None` with `ge=1`.
- **Production use:** When > 1, passed as `--tensor-parallel-size` (`cmd/server/main.py:229-231`).
- **Test:** `test_manifest_endpoint::TestBuildVllmCmd::test_tensor_parallel_size_present/absent`

### `runtime.serving_engine.pipeline_parallel_size`
- **Effect:** validated, passed-to-vllm
- **Verification:** Pydantic `int | None` with `ge=1`.
- **Production use:** When > 1, passed as `--pipeline-parallel-size` (`cmd/server/main.py:233-235`).
- **Test:** `test_manifest_endpoint::TestBuildVllmCmd::test_pipeline_parallel_size_present/absent`

### `runtime.serving_engine.disable_custom_all_reduce`
- **Effect:** validated, passed-to-vllm
- **Verification:** Pydantic `bool | None`.
- **Production use:** When true, adds `--disable-custom-all-reduce` (`cmd/server/main.py:237-238`).
- **Test:** `test_manifest_endpoint::TestBuildVllmCmd::test_disable_custom_all_reduce_present/absent`

---

## `hardware_profile`

### `hardware_profile.gpu.model`
- **Effect:** validated, compared
- **Verification:** Pydantic `str` with `min_length=1`.
- **Production use:** Compared against actual GPU name (case-insensitive substring match) (`cmd/server/main.py:128-129`).
- **Test:** `test_manifest_endpoint::TestCheckHardware::test_mismatch_gpu_model`

### `hardware_profile.gpu.count`
- **Effect:** validated, compared
- **Verification:** Pydantic `int` with `ge=1`.
- **Production use:** Compared against `torch.cuda.device_count()` (`cmd/server/main.py:125-126`).
- **Test:** `test_manifest_endpoint::TestCheckHardware::test_mismatch_gpu_count`

### `hardware_profile.gpu.driver_version`
- **Effect:** validated, compared
- **Verification:** Pydantic `str` with `min_length=1`.
- **Production use:** Compared against actual NVIDIA driver version (`cmd/server/main.py:131-134`).
- **Test:** `test_manifest_endpoint::TestCheckHardware::test_mismatch_driver_version`, `test_unqueryable_driver`

### `hardware_profile.gpu.cuda_driver_version`
- **Effect:** validated, compared
- **Verification:** Pydantic `str` with `min_length=1`.
- **Production use:** Compared against `torch.version.cuda` (`cmd/server/main.py:136-137`).
- **Test:** `test_manifest_endpoint::TestCheckHardware::test_mismatch_cuda_version`

---

## `requests[]`

### `requests[].id`
- **Effect:** validated, used-in-computation, recorded
- **Verification:** Pydantic pattern `^[A-Za-z0-9._:-]+$`.
- **Production use:** Seeds per-request RNG (`cmd/runner/main.py:281,285`). Used as request tracking ID in vLLM chat completion (`cmd/server/main.py:675`). Recorded in capture output.
- **Test:** `test_manifest_model::TestSubModels::test_request_item`

### `requests[].prompt`
- **Effect:** validated, passed-to-vllm
- **Verification:** Pydantic `str` with `min_length=1`.
- **Production use:** Passed as chat message content to vLLM (`cmd/server/main.py:678`). Passed to vLLM `generate()` (`cmd/runner/vllm_runner.py:97`).
- **Test:** `test_manifest_endpoint::TestRequestValidation`

### `requests[].max_new_tokens`
- **Effect:** validated, passed-to-vllm, used-in-computation
- **Verification:** Pydantic `int` with `ge=1`.
- **Production use:** Validated against `max_model_len` (`cmd/server/main.py:154`). Passed as `max_tokens` to vLLM (`cmd/server/main.py:679`, `cmd/runner/vllm_runner.py:102`).
- **Test:** `test_manifest_endpoint::TestRequestValidation::test_request_exceeding_max_model_len_rejected`

### `requests[].temperature`
- **Effect:** validated, passed-to-vllm
- **Verification:** Pydantic `float` with `ge=0, le=2`.
- **Production use:** Passed to vLLM in chat completion body (`cmd/server/main.py:680`). Passed to `SamplingParams` (`cmd/runner/vllm_runner.py:101`).
- **Test:** `test_manifest_endpoint::TestSchemaValidation::test_valid_manifest_passes` (uses temperature=0). No boundary test.
- **Gap:** No test for out-of-range temperature rejection at Pydantic level.

---

## `comparison`

### `comparison.tokens.mode` / `comparison.logits.mode`
- **Effect:** validated, controls-branch
- **Verification:** Pydantic `ComparisonMode` enum (exact, ulp, absrel, hash).
- **Production use:** Selects comparison algorithm in verifier (`cmd/verifier/main.py:137-138`). The full `Comparator` (including mode-specific params) is passed to `_compare_observable()`.
- **Test:** Enum coverage in `test_manifest_field_coverage`. Behavioral coverage in `tests/e2e/test_verifier_outputs.py`.

### `comparison.tokens.algorithm` / `comparison.logits.algorithm`
- **Effect:** validated, used-in-computation
- **Verification:** Pydantic `str | None`. Schema requires it when mode=hash.
- **Production use:** Passed to `_compare_observable()` via `model_dump()` (`cmd/verifier/main.py:135-136`).
- **Test:** Field coverage test only. No behavioral test.
- **Gap:** No test verifies hash-mode comparison actually uses the algorithm field.

### `comparison.tokens.ulp` / `comparison.logits.ulp`
- **Effect:** validated, used-in-computation
- **Verification:** Pydantic `int | None` with `ge=0`. Schema requires it when mode=ulp.
- **Production use:** Passed to `_compare_observable()` for ULP comparison.
- **Test:** `test_spec_review_bugs::TestBug3UlpComparisonWrong` (behavioral). Field coverage test.

### `comparison.tokens.atol` / `comparison.logits.atol`
### `comparison.tokens.rtol` / `comparison.logits.rtol`
- **Effect:** validated, used-in-computation
- **Verification:** Pydantic `float | None` with `ge=0`. Schema requires both when mode=absrel.
- **Production use:** Passed to `_compare_observable()` for absolute+relative tolerance comparison.
- **Test:** `test_manifest_model::TestSubModels::test_comparator_absrel`. Field coverage test.
- **Gap:** No behavioral test verifying atol/rtol values affect comparison outcome.

---

## `artifact_inputs[]`

### `artifact_inputs[].artifact_id`
- **Effect:** validated, recorded, used-in-computation
- **Verification:** Pydantic pattern `^[A-Za-z0-9._:-]+$`.
- **Production use:** Used as identifier in lockfile artifacts (`cmd/resolver/main.py:44`). Cross-checked against lockfile in runner (`cmd/runner/main.py:337-340`). Used in builder descriptor (`cmd/builder/main.py`).
- **Test:** Field coverage test. Integration tests via resolver/builder.

### `artifact_inputs[].artifact_type`
- **Effect:** validated, controls-branch, used-in-computation
- **Verification:** Pydantic `ArtifactType` enum (19 values).
- **Production use:** Used to filter model artifacts for HF resolution (`cmd/resolver/main.py:30-37`). Checked against `model_types` set for file verification (`cmd/server/main.py:326-328`). Used to categorize builder components (`cmd/builder/main.py`).
- **Test:** All 19 values covered in field coverage test. Behavioral coverage in resolver/builder integration tests.

### `artifact_inputs[].source_kind`
- **Effect:** validated, recorded
- **Verification:** Pydantic `SourceKind` enum (7 values).
- **Production use:** Recorded in lockfile artifact (`cmd/resolver/main.py:47`). Checked for OCI artifacts in builder (`cmd/builder/main.py:86`).
- **Test:** All 7 values covered in field coverage test.
- **Gap:** No behavioral test that `source_kind` affects builder/resolver behavior differently per kind.

### `artifact_inputs[].source_uri`
- **Effect:** validated, recorded
- **Verification:** Pydantic `str` (no pattern constraint).
- **Production use:** Recorded as `uri` in lockfile artifact (`cmd/resolver/main.py:48`). Used in builder descriptor (`cmd/builder/main.py`).
- **Test:** Field coverage test only.

### `artifact_inputs[].immutable_ref`
- **Effect:** validated, recorded, used-in-computation
- **Verification:** Pydantic `str` with `min_length=1`.
- **Production use:** Recorded in lockfile artifact (`cmd/resolver/main.py:49`). Used in builder for OCI validation and descriptor (`cmd/builder/main.py`). Used as sort key for deterministic artifact ordering.
- **Test:** Field coverage test. Integration tests via resolver/builder.

### `artifact_inputs[].name`
- **Effect:** validated, recorded
- **Verification:** Pydantic `str | None`.
- **Production use:** Used as artifact name in lockfile, defaults to `artifact_id` (`cmd/resolver/main.py:46`).
- **Test:** Field coverage test only.

### `artifact_inputs[].expected_digest`
- **Effect:** validated, compared, used-in-computation
- **Verification:** Pydantic pattern `^sha256:[a-f0-9]{64}$`.
- **Production use:** Used as lockfile digest if present, otherwise computed (`cmd/resolver/main.py:42`). Verified against actual file hash in HF cache (`cmd/server/main.py:330-354`).
- **Test:** `test_manifest_endpoint::TestVerifyModelArtifacts` (correct, wrong, size mismatch, missing file)

### `artifact_inputs[].size_bytes`
- **Effect:** validated, compared, recorded
- **Verification:** Pydantic `int | None` with `ge=1`.
- **Production use:** Quick size check before full digest verification (`cmd/server/main.py:341-347`). Recorded in lockfile, summed for closure size in builder (`cmd/builder/main.py:130`).
- **Test:** `test_manifest_endpoint::TestVerifyModelArtifacts::test_size_mismatch_raises`

### `artifact_inputs[].path`
- **Effect:** validated, used-in-computation
- **Verification:** Pydantic `str | None` with `min_length=1`.
- **Production use:** Used to locate file in HF cache for digest verification (`cmd/server/main.py:331-335`).
- **Test:** `test_manifest_endpoint::TestVerifyModelArtifacts` (missing file test)

### `artifact_inputs[].role`
- **Effect:** DEAD
- **Verification:** Pydantic `FileRole | None` enum (6 values).
- **Production use:** **Never read by any production code.** Parsed and validated by Pydantic, then discarded. Not referenced anywhere in `cmd/` or `pkg/` (verified by grep).
- **Test:** Field coverage test only.
- **Issue:** This field is validated but has zero runtime effect. Either it should be used (e.g., in artifact verification or builder categorization) or removed from the schema.

---

## Summary of Gaps

| Gap | Fields affected | Severity |
|---|---|---|
| **DEAD field** — parsed but never read | `artifact_inputs[].role` | Medium: schema bloat, false sense of coverage |
| **Hardcoded override** — manifest value ignored in runner | `cublas_workspace_config`, `pythonhashseed` | High: user sets value in manifest but runner ignores it |
| **No behavioral test** for vLLM flag | `dtype`, `attention_backend`, `max_num_seqs`, `gpu_memory_utilization`, `enforce_eager`, `trust_remote_code`, `tokenizer_revision` | Medium: rely on code reading only, no test proves the flag reaches vLLM |
| **No behavioral test** for env var | `cuda_launch_blocking`, `batch_invariance.enabled`, `torch_deterministic` | Medium: critical determinism knobs with no test proving they take effect |
| **No behavioral test** for comparison params | `algorithm` (hash mode), `atol`/`rtol` (absrel mode) | Medium: verifier uses them but no test proves changing the value changes the outcome |
| **No behavioral test** for `created_at` | `created_at` | Low: used as lockfile timestamp but never tested |
| **No boundary test** for `temperature` | `requests[].temperature` | Low: Pydantic enforces range but no rejection test |
