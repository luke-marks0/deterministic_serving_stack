# Plan: Replace manifest dicts with Pydantic models

## Goal

Replace `dict[str, Any]` manifest passing throughout the codebase with typed Pydantic models. The Pydantic model becomes the source of truth — the JSON Schema file is generated from it.

## Why

Current code:
```python
manifest["runtime"]["serving_engine"]["max_model_len"]  # KeyError if typo
```

After:
```python
manifest.runtime.serving_engine.max_model_len  # AttributeError at import time, IDE catches it
```

Benefits:
- IDE autocomplete everywhere
- Typos caught statically
- One source of truth (Pydantic model, not hand-maintained JSON Schema)
- Validation built into parsing — no separate `validate_with_schema` call
- Serialization for free (`model.model_dump()`, `model.model_dump_json()`)

## Dependencies

Pydantic is already installed (vLLM depends on it). No new packages needed.

## The model

Create `pkg/manifest/model.py`:

```python
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# -- Enums --

class AttentionBackend(str, Enum):
    FLASH_ATTN = "FLASH_ATTN"
    TRITON_ATTN = "TRITON_ATTN"
    FLASH_ATTN_MLA = "FLASH_ATTN_MLA"
    TRITON_MLA = "TRITON_MLA"

class Dtype(str, Enum):
    auto = "auto"
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"

class ComparisonMode(str, Enum):
    exact = "exact"
    ulp = "ulp"
    absrel = "absrel"
    hash = "hash"

class ArtifactType(str, Enum):
    model_weights = "model_weights"
    model_config = "model_config"
    tokenizer = "tokenizer"
    generation_config = "generation_config"
    chat_template = "chat_template"
    prompt_formatter = "prompt_formatter"
    serving_stack = "serving_stack"
    container_image = "container_image"
    cuda_lib = "cuda_lib"
    kernel_library = "kernel_library"
    runtime_knob_set = "runtime_knob_set"
    request_set = "request_set"
    compiled_extension = "compiled_extension"
    remote_code = "remote_code"
    collective_stack = "collective_stack"

class SourceKind(str, Enum):
    hf = "hf"
    oci = "oci"
    s3 = "s3"
    http = "http"
    git = "git"
    nix = "nix"
    inline = "inline"

class FileRole(str, Enum):
    weights_shard = "weights_shard"
    config = "config"
    tokenizer = "tokenizer"
    generation_config = "generation_config"
    chat_template = "chat_template"
    prompt_formatter = "prompt_formatter"


# -- Sub-models --

class GpuProfile(BaseModel):
    model: str = Field(min_length=1)
    count: int = Field(ge=1)
    driver_version: str = Field(min_length=1)
    cuda_driver_version: str = Field(min_length=1)

class HardwareProfile(BaseModel):
    gpu: GpuProfile

class ServingEngine(BaseModel):
    max_model_len: int = Field(ge=1)
    max_num_seqs: int = Field(ge=1)
    gpu_memory_utilization: float = Field(ge=0.1, le=1.0)
    dtype: Dtype = Dtype.auto
    attention_backend: AttentionBackend
    # Optional fields — when absent, vLLM uses its own defaults
    quantization: str | None = None
    load_format: str | None = None
    kv_cache_dtype: str | None = None
    max_num_batched_tokens: int | None = None
    block_size: int | None = None
    enable_prefix_caching: bool | None = None
    enable_chunked_prefill: bool | None = None
    scheduling_policy: str | None = None
    disable_sliding_window: bool | None = None
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    disable_custom_all_reduce: bool | None = None

class BatchInvariance(BaseModel):
    enabled: bool
    enforce_eager: bool

class DeterministicKnobs(BaseModel):
    seed: int = Field(ge=0)
    torch_deterministic: bool
    cuda_launch_blocking: bool
    cublas_workspace_config: str = ":4096:8"
    pythonhashseed: str = "0"

class RuntimeConfig(BaseModel):
    strict_hardware: bool
    batch_invariance: BatchInvariance
    deterministic_knobs: DeterministicKnobs
    serving_engine: ServingEngine
    container_image_digest: str | None = Field(
        default=None, pattern=r"^sha256:[a-f0-9]{64}$"
    )

class ModelConfig(BaseModel):
    source: str = Field(pattern=r"^hf://[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
    weights_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    tokenizer_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    trust_remote_code: bool

class Request(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9._:-]+$")
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(ge=1, le=4096)
    temperature: float = Field(ge=0, le=2)

class Comparator(BaseModel):
    mode: ComparisonMode
    algorithm: str | None = None  # for hash mode
    ulp: int | None = None        # for ulp mode
    atol: float | None = None     # for absrel mode
    rtol: float | None = None     # for absrel mode

class ComparisonConfig(BaseModel):
    tokens: Comparator
    logits: Comparator

class ArtifactInput(BaseModel):
    artifact_id: str = Field(pattern=r"^[A-Za-z0-9._:-]+$")
    artifact_type: ArtifactType
    source_kind: SourceKind
    source_uri: str
    immutable_ref: str = Field(min_length=1)
    # Optional fields
    name: str | None = None
    expected_digest: str | None = Field(default=None, pattern=r"^sha256:[a-f0-9]{64}$")
    size_bytes: int | None = Field(default=None, ge=1)
    path: str | None = None
    role: FileRole | None = None


# -- Top-level manifest --

class Manifest(BaseModel):
    manifest_version: Literal["v1"] = "v1"
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
    created_at: str  # ISO 8601 datetime
    model: ModelConfig
    runtime: RuntimeConfig
    hardware_profile: HardwareProfile
    requests: list[Request] = Field(min_length=1)
    comparison: ComparisonConfig
    artifact_inputs: list[ArtifactInput] = Field(min_length=4)
```

## Implementation plan

### Task 1: Create the Pydantic model

**File:** `pkg/manifest/model.py`

Write the model above. Then verify it parses the real manifest:

```python
import json
from pkg.manifest.model import Manifest

data = json.load(open("manifests/qwen3-1.7b.manifest.json"))
m = Manifest.model_validate(data)
print(m.model.source)                    # hf://Qwen/Qwen3-1.7B
print(m.runtime.serving_engine.dtype)    # auto
print(m.requests[0].prompt)              # Hello
```

Write a unit test that loads the real manifest and the positive fixture and validates both parse successfully.

**Commit:** `feat: add Pydantic manifest model`

### Task 2: Generate JSON Schema from the model

**File:** `scripts/ci/generate_schema.py`

```python
import json
from pkg.manifest.model import Manifest

schema = Manifest.model_json_schema()
print(json.dumps(schema, separators=(",", ":"), sort_keys=True))
```

Compare the generated schema against the hand-maintained one. They won't match exactly (Pydantic's schema format differs from our hand-written one), but the validation behavior should be equivalent. Test: any JSON that passes the old schema should also parse into the Pydantic model, and vice versa.

For now, keep both. The hand-maintained schema stays for the CI gate. The Pydantic model is used in Python code. Once we're confident they're equivalent, delete the hand-maintained schema and generate it in CI.

**Commit:** `feat: add schema generation from Pydantic model`

### Task 3: Migrate the server

**File:** `cmd/server/main.py`

This is the main consumer. Change:

```python
# Before
def _build_vllm_cmd(manifest: dict[str, Any], host: str, port: int) -> list[str]:
    model_source = manifest["model"]["source"]
    ...

# After
def _build_vllm_cmd(manifest: Manifest, host: str, port: int) -> list[str]:
    model_source = manifest.model.source
    ...
```

Specific changes:
- `_build_vllm_cmd(manifest: dict → Manifest)`
- `_enforce_hardware(manifest: dict → Manifest)`
- `_enforce_model_revision(manifest: dict → Manifest)`
- `_validate_requests(manifest: dict → Manifest)`
- `_set_deterministic_env(manifest: dict → Manifest)`
- `_start_vllm(state, manifest: dict → Manifest)`
- `_verify_container_image(manifest: dict → Manifest)`
- `_verify_model_artifacts(manifest: dict → Manifest)`
- `ServerState.manifest: dict → Manifest`
- `_handle_post_manifest`: parse with `Manifest.model_validate(body)` instead of `validate_with_schema()`
- `_handle_get_manifest`: use `state.manifest.model_dump()` for JSON serialization
- `manifest_digest` property: `canonical_json_bytes(manifest.model_dump())`

The key refactoring pattern:
```python
# Before
manifest["runtime"]["serving_engine"]["max_model_len"]
manifest["model"].get("trust_remote_code", False)
manifest.get("runtime", {}).get("container_image_digest")

# After
manifest.runtime.serving_engine.max_model_len
manifest.model.trust_remote_code
manifest.runtime.container_image_digest
```

Every `manifest[` becomes `manifest.` — grep for `manifest\[` to find them all.

**Test:** Update `tests/unit/test_manifest_endpoint.py` — pass `Manifest` objects instead of dicts to the functions. The schema validation tests can use `Manifest.model_validate()` instead of `validate_with_schema()`.

**Commit:** `refactor: migrate server to Pydantic manifest model`

### Task 4: Migrate the resolver

**File:** `cmd/resolver/main.py`, `pkg/common/hf_resolution.py`

The resolver is trickier because it **mutates** the manifest (writes `weights_revision`, `tokenizer_revision`, merges `artifact_inputs`). With Pydantic, mutation works via:

```python
# Option A: model_copy with update
manifest = manifest.model_copy(update={
    "model": manifest.model.model_copy(update={
        "weights_revision": resolved.resolved_revision,
        "tokenizer_revision": resolved.resolved_revision,
    })
})

# Option B: construct a new Manifest
# More verbose but explicit
```

Or make the model mutable during resolution by using `model_config = ConfigDict(frozen=False)` on a resolver-specific subclass. Then freeze it when done.

The resolver's input might be a partial manifest (pre-resolution, some fields not yet filled). Handle this by either:
- Making `weights_revision` and `tokenizer_revision` optional with defaults
- Having a separate `UnresolvedManifest` model with optional revision fields
- Keeping the resolver on dicts and only converting to `Manifest` after resolution

**Recommendation:** Keep the resolver on dicts. It's the only code that mutates the manifest. Convert to `Manifest` at the boundary (after resolution, before handing to the server). This avoids the partial-model problem.

```python
# cmd/resolver/main.py
def resolve_manifest_to_lockfile(manifest_dict: dict, ...) -> dict:
    # ... mutate manifest_dict as before ...
    # Validate at the end:
    Manifest.model_validate(manifest_dict)
    return lockfile
```

**Commit:** `refactor: validate resolved manifest with Pydantic model`

### Task 5: Migrate the runner, capture, and verifier

**Files:** `cmd/runner/main.py`, `cmd/runner/vllm_runner.py`, `cmd/capture/main.py`, `cmd/verifier/main.py`

Same pattern as Task 3 — change function signatures from `dict[str, Any]` to `Manifest`, replace bracket access with dot access.

The runner and capture generate run bundles, which are separate from the manifest model. Don't try to model the run bundle as Pydantic yet — that's a separate task.

The verifier reads `manifest["comparison"]` — change to `manifest.comparison.tokens.mode`, etc.

**Commit:** `refactor: migrate runner, capture, verifier to Pydantic manifest model`

### Task 6: Delete hand-maintained schema (optional)

Once Tasks 1-5 are done and CI is green:

1. Replace `schemas/manifest.v1.schema.json` with a generated version:
   ```bash
   python3 scripts/ci/generate_schema.py > schemas/manifest.v1.schema.json
   ```
2. Update `scripts/ci/schema_gate.sh` to regenerate and diff (ensuring the checked-in schema matches the model)
3. Delete `validate_with_schema("manifest.v1.schema.json", ...)` calls — Pydantic does this

**Caveat:** The generated Pydantic JSON Schema may differ in structure from our hand-written one. The fixture validation in CI validates positive/negative fixtures against the schema. You'll need to verify that the generated schema still correctly rejects the negative fixtures.

If the schemas aren't equivalent (Pydantic's conditionals work differently than JSON Schema's `allOf`/`if-then`), keep both: Pydantic for Python code, JSON Schema for CI fixture validation. This is fine — they serve different purposes.

**Commit:** `refactor: generate manifest schema from Pydantic model`

## Commit sequence

```
1. feat: add Pydantic manifest model
2. feat: add schema generation from Pydantic model
3. refactor: migrate server to Pydantic manifest model
4. refactor: validate resolved manifest with Pydantic model
5. refactor: migrate runner, capture, verifier to Pydantic manifest model
6. refactor: generate manifest schema from Pydantic model (optional)
```

Tasks 1-2 are additive (no breaking changes). Tasks 3-5 are the migration. Task 6 is cleanup.

## Testing strategy

- **Task 1:** Test that the real manifest and fixture parse successfully. Test that invalid data raises `ValidationError`.
- **Tasks 3-5:** Existing tests should keep passing — you're changing internal types, not external behavior. If a test constructs a manifest dict, either convert it to `Manifest(...)` or use `Manifest.model_validate(dict)`.
- **Task 6:** Verify generated schema rejects negative fixtures.

## What NOT to model

- **Lockfile** — separate schema, separate lifecycle. Keep as dict for now.
- **Run bundle** — generated output, not input. Keep as dict.
- **Verify report** — output of verifier. Keep as dict.

Only the manifest benefits from typing because it's the primary input that flows through every component.
