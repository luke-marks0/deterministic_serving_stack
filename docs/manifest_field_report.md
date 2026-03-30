# Manifest Field Report

Every field in `manifest.v1.schema.json` and how it's used across the codebase.

## Top-level


| Field              | Type         | Used in                                                | How                                                                                                |
| ------------------ | ------------ | ------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| `manifest_version` | `const "v1"` | Schema validation only                                 | Ensures forward compatibility                                                                      |
| `run_id`           | `string`     | `cmd/server/main.py:421`, `cmd/runner/main.py:313,484` | Identifies the run in responses, seeds per-request hashes, appears in boot records and run bundles |
| `created_at`       | `date-time`  | `cmd/resolver/main.py:150`                             | Carried into lockfile as the deterministic timestamp                                               |


## model


| Field                      | Used in                                                                                                      | How                                                                                                                                                                                |
| -------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model.source`             | `cmd/server/main.py:191` → `--model` flag, `cmd/runner/vllm_runner.py:37`, `cmd/resolver/main.py:99`         | **Enforced.** Passed to vLLM as the model to load. Resolver uses it to look up HF repo.                                                                                            |
| `model.resolved_revision`  | `cmd/server/main.py:164` → `--revision` flag, `cmd/resolver/main.py:114`                                     | **Enforced.** Pins vLLM to the exact HF commit. Resolver writes it after resolution.                                                                                               |
| `model.tokenizer_revision` | `cmd/server/main.py:206` → `--tokenizer-revision` flag, `cmd/resolver/main.py:116`                           | **Enforced.** Ensures tokenizer matches the pinned model version.                                                                                                                  |
| `model.weights_revision`   | `cmd/resolver/main.py:115`                                                                                   | Written by resolver. Not separately passed to vLLM (same as resolved_revision).                                                                                                    |
| `model.trust_remote_code`  | `cmd/server/main.py:233` → `--trust-remote-code`, `cmd/runner/vllm_runner.py:75`, `cmd/resolver/main.py:106` | **Enforced.** Controls whether vLLM loads custom model code. Schema requires `remote_code` block when true.                                                                        |
| `model.remote_code`        | `cmd/resolver/main.py:119`                                                                                   | Written by resolver with commit SHA + digest when `trust_remote_code=true`. Not enforced at serve time (vLLM handles it).                                                          |
| `model.required_files`     | `cmd/resolver/main.py:117`                                                                                   | Written by resolver with per-file role, path, URI, sha256 digest, size. **Not verified at serve time** — vLLM downloads files itself. Would need a pre-download verification step. |
| `model.requested_revision` | Schema only                                                                                                  | Optional user-facing hint; resolver replaces with `resolved_revision`.                                                                                                             |


## runtime

### runtime.serving_engine


| Field                                   | Used in                                                                        | How                                                                                       |
| --------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| `serving_engine.max_model_len`          | `cmd/server/main.py:214` → `--max-model-len`, `main.py:174` request validation | **Enforced.** Sets context window. Requests exceeding it are rejected before vLLM starts. |
| `serving_engine.max_num_seqs`           | `cmd/server/main.py:220` → `--max-num-seqs`                                    | **Enforced.** Limits concurrent sequences in the engine.                                  |
| `serving_engine.gpu_memory_utilization` | `cmd/server/main.py:217` → `--gpu-memory-utilization`                          | **Enforced.** Fraction of GPU memory for KV cache.                                        |
| `serving_engine.dtype`                  | `cmd/server/main.py:211` → `--dtype`                                           | **Enforced.** Model precision (auto/float16/bfloat16/float32).                            |
| `serving_engine.attention_backend`      | `cmd/server/main.py:224` → `--attention-backend`                               | **Enforced.** Which attention kernel to use (FLASH_ATTN/TRITON_ATTN/etc).                 |


### runtime.batch_invariance


| Field                            | Used in                                                 | How                                                                 |
| -------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------- |
| `batch_invariance.enabled`       | `cmd/server/main.py:279` → `VLLM_BATCH_INVARIANT=1` env | **Enforced.** Enables vLLM's batch-invariant mode.                  |
| `batch_invariance.enforce_eager` | `cmd/server/main.py:229` → `--enforce-eager`            | **Enforced.** Disables CUDA graphs (required for batch invariance). |


### runtime.deterministic_knobs


| Field                                      | Used in                                                       | How                                                                                                            |
| ------------------------------------------ | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `deterministic_knobs.seed`                 | `cmd/server/main.py:198` → `--seed`, `cmd/runner/main.py:313` | **Enforced.** Global seed for vLLM engine. Runner also derives per-request seeds from it.                      |
| `deterministic_knobs.torch_deterministic`  | `cmd/runner/vllm_runner.py:62`                                | **Enforced in runner.** Sets `torch.use_deterministic_algorithms(True)`. Server relies on vLLM's own handling. |
| `deterministic_knobs.cuda_launch_blocking` | `cmd/server/main.py:275` → `CUDA_LAUNCH_BLOCKING` env         | **Enforced.** Synchronous CUDA launches.                                                                       |


### runtime.batch_cardinality


| Field                                 | Used in                                                 | How                                                                                                                           |
| ------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `batch_cardinality.target_batch_size` | `cmd/runner/main.py:306,446`, `cmd/capture/main.py:140` | **Used in runner/capture.** Controls how requests are batched. Not directly enforced by the server (vLLM handles scheduling). |
| `batch_cardinality.min_requests`      | Schema only                                             | Declares minimum batch size.                                                                                                  |
| `batch_cardinality.max_requests`      | Schema only                                             | Declares maximum batch size.                                                                                                  |


### runtime.engine_trace


| Field                  | Used in                      | How                                                                                                                          |
| ---------------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `engine_trace.enabled` | `cmd/runner/main.py:324-329` | **Used in runner.** Controls which engine events are logged (batch composition, reordering, attention/collective selection). |
| `engine_trace.events`  | `cmd/runner/main.py:324-329` | **Used in runner.** Each event type adds corresponding entries to the run bundle.                                            |


### runtime.nix_pin


| Field                | Used in                | How                                                                                                           |
| -------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------- |
| `nix_pin.flake_ref`  | Not currently consumed | **Not enforced.** Declares which Nix flake built the closure. Should be verified against the running closure. |
| `nix_pin.flake_hash` | Not currently consumed | **Not enforced.** SHA256 of the flake inputs.                                                                 |


### Other runtime fields


| Field                                   | Used in                                                | How                                                                                   |
| --------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| `runtime.strict_hardware`               | `cmd/server/main.py:300`, `cmd/runner/main.py:425,432` | **Enforced.** If true, hardware mismatches are fatal errors. If false, warnings only. |
| `runtime.batch_policy`                  | `cmd/capture/main.py:271`, `cmd/runner/main.py:532`    | **Used in capture/runner.** Recorded in run bundle metadata (fixed/queued_fixed).     |
| `runtime.allow_non_reproducible_egress` | Schema conditional                                     | Required to be true when `network.security_mode=tls`.                                 |


## hardware_profile

### hardware_profile.gpu


| Field                     | Used in                                                   | How                                                                                                     |
| ------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `gpu.vendor`              | Schema only                                               | Must be "nvidia".                                                                                       |
| `gpu.model`               | `cmd/server/main.py:145`, `cmd/runner/vllm_runner.py:191` | **Enforced.** Server validates GPU name matches (substring). Runner uses as fallback for GPU inventory. |
| `gpu.count`               | `cmd/server/main.py:139`                                  | **Enforced.** Server validates device count matches.                                                    |
| `gpu.driver_version`      | `cmd/runner/vllm_runner.py:177`                           | **Used in runner.** Recorded in run bundle env_info.                                                    |
| `gpu.cuda_driver_version` | Schema only                                               | Declared but not checked at runtime.                                                                    |
| `gpu.pci_ids`             | Schema only                                               | Declared but not checked at runtime.                                                                    |


### hardware_profile.nic


| Field                 | Used in                  | How                                                                    |
| --------------------- | ------------------------ | ---------------------------------------------------------------------- |
| `nic.model`           | `cmd/runner/main.py:420` | **Used in runner.** Hardware conformance check validates NIC presence. |
| `nic.pci_id`          | `cmd/runner/main.py:420` | **Used in runner.** Validated against system NICs.                     |
| `nic.firmware`        | Schema only              | Declared but not verified.                                             |
| `nic.link_speed_gbps` | Schema only              | Declared but not verified.                                             |
| `nic.offloads`        | Schema only              | TSO/GSO/checksum/VLAN offload declarations. Not enforced.              |


### hardware_profile.topology


| Field                        | Used in                                                                       | How                                                                                                                                                |
| ---------------------------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `topology.mode`              | `cmd/server/main.py:150`, `cmd/runner/main.py:329`, `cmd/resolver/main.py:40` | **Enforced.** Server validates against GPU count. Runner uses to determine collective algorithm. Resolver validates collective stack requirements. |
| `topology.node_count`        | `cmd/server/main.py:151`, `cmd/runner/dispatcher.py:23`                       | **Enforced.** Validated against available GPUs.                                                                                                    |
| `topology.rack_count`        | `cmd/runner/dispatcher.py:23`                                                 | **Used in runner.** Dispatcher uses for rack-aware scheduling.                                                                                     |
| `topology.collective_fabric` | `cmd/runner/main.py:329`                                                      | **Used in runner.** Determines collective algorithm selection events.                                                                              |


## network


| Field                              | Used in                  | How                                                                                                                 |
| ---------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| `network.security_mode`            | `cmd/runner/main.py:554` | **Used in runner.** Recorded in run bundle. Schema enforces: if `tls`, then `egress_reproducibility` must be false. |
| `network.egress_reproducibility`   | Schema conditional       | Tied to security_mode constraint.                                                                                   |
| `network.mtu`                      | Schema only              | Declared for deterministic framing. Not enforced at runtime.                                                        |
| `network.mss`                      | Schema only              | Declared for deterministic framing.                                                                                 |
| `network.tso/gso/checksum_offload` | Schema only              | NIC offload declarations.                                                                                           |
| `network.queue_mapping`            | Schema only              | TX/RX queue mapping policy.                                                                                         |
| `network.ring_sizes`               | Schema only              | Ring buffer sizes.                                                                                                  |
| `network.thread_affinity`          | Schema only              | CPU core pinning for network threads.                                                                               |
| `network.internal_batching`        | Schema only              | Packet batching config.                                                                                             |


## requests


| Field                       | Used in                                                              | How                                                                                                             |
| --------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `requests[].id`             | `cmd/runner/main.py:312,313`, `cmd/server/main.py:176`               | **Enforced.** Server validates token limits. Runner uses ID for per-request seed derivation and bundle entries. |
| `requests[].prompt`         | `cmd/runner/main.py:313`, `cmd/runner/vllm_runner.py:96`             | **Used in runner.** The actual text sent to vLLM for inference.                                                 |
| `requests[].max_new_tokens` | `cmd/server/main.py:177` validation, `cmd/runner/vllm_runner.py:100` | **Enforced.** Server rejects if > max_model_len. Runner passes to SamplingParams.                               |
| `requests[].temperature`    | `cmd/runner/vllm_runner.py:99`                                       | **Used in runner.** Passed to SamplingParams.                                                                   |


## comparison


| Field                       | Used in                    | How                                                                                                               |
| --------------------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `comparison.tokens`         | `cmd/verifier/main.py:140` | **Used in verifier.** Comparator mode (exact/ulp/absrel/hash) for token-level comparison between two run bundles. |
| `comparison.logits`         | `cmd/verifier/main.py:140` | **Used in verifier.** Comparator for logit values.                                                                |
| `comparison.activations`    | `cmd/verifier/main.py:140` | **Used in verifier.** Comparator for activation values.                                                           |
| `comparison.network_egress` | `cmd/verifier/main.py:140` | **Used in verifier.** Comparator for network egress bytes.                                                        |


Surfaced in `GET /manifest` response as `active_config.comparison_modes` (`cmd/server/main.py:484`).

## deterministic_dispatcher


| Field                                           | Used in                                                      | How                                                                                                            |
| ----------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `deterministic_dispatcher.enabled`              | Schema conditional                                           | Required when topology is replicated/TP/PP.                                                                    |
| `deterministic_dispatcher.algorithm`            | `cmd/coordinator/main.py:101`, `cmd/runner/dispatcher.py:24` | **Used in coordinator/runner.** Determines how requests are assigned to nodes (round_robin_hash/sequence_map). |
| `deterministic_dispatcher.request_order_source` | Schema only                                                  | Declares ordering source (ingress_sequence).                                                                   |
| `deterministic_dispatcher.replay_log_required`  | Schema only                                                  | Whether dispatch log must be saved.                                                                            |


## artifact_inputs


| Field                               | Used in                                                  | How                                                                                                                           |
| ----------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `artifact_inputs[].artifact_id`     | `cmd/runner/main.py:408`, `cmd/resolver/main.py:123,152` | **Used in runner/resolver.** Runner builds artifact list for run bundle. Resolver merges model artifacts after HF resolution. |
| `artifact_inputs[].artifact_type`   | `cmd/resolver/main.py:48`                                | **Used in resolver.** Validates collective_stack presence for multi-node topologies.                                          |
| `artifact_inputs[].source_kind`     | `cmd/resolver/main.py:152`                               | **Used in resolver.** Carried into lockfile artifacts.                                                                        |
| `artifact_inputs[].source_uri`      | `cmd/resolver/main.py:152`                               | **Used in resolver.** Carried into lockfile.                                                                                  |
| `artifact_inputs[].immutable_ref`   | `cmd/resolver/main.py:152`                               | **Used in resolver.** Pinned reference for reproducibility.                                                                   |
| `artifact_inputs[].expected_digest` | Not currently verified                                   | **Not enforced.** Declared sha256 digest of the artifact. Should be verified against actual content.                          |


Count surfaced in `GET /manifest` response as `active_config.artifact_inputs` (`cmd/server/main.py:483`).

## Enforcement gaps

Fields that are declared in the manifest but **not currently enforced** at runtime:


| Field                               | Gap                                                | Risk                                     |
| ----------------------------------- | -------------------------------------------------- | ---------------------------------------- |
| `model.required_files[].digest`     | Per-file SHA256 not verified after download        | Could load tampered weights              |
| `runtime.nix_pin`                   | Flake ref/hash not checked against running closure | Could run wrong software version         |
| `artifact_inputs[].expected_digest` | Artifact digests not verified                      | Could use wrong versions of dependencies |
| `hardware_profile.gpu.pci_ids`      | PCI slot IDs not checked                           | Low risk — GPU model/count is checked    |
| `hardware_profile.nic.firmware`     | NIC firmware not verified                          | Could affect network determinism         |
| `network.`* (most fields)           | No userspace networking stack running              | L2 determinism is scaffolding only       |


