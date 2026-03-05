from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Protocol

from pkg.common.deterministic import sha256_file


class HFResolutionError(Exception):
    pass


class HFClient(Protocol):
    def resolve_commit(self, repo_id: str, revision: str) -> str:
        ...

    def list_files(self, repo_id: str, revision: str) -> list[str]:
        ...

    def download_file(self, repo_id: str, revision: str, file_path: str, cache_dir: Path | None) -> Path:
        ...


@dataclass(frozen=True)
class ResolvedHF:
    resolved_revision: str
    required_files: list[dict[str, Any]]
    model_artifacts: list[dict[str, Any]]
    remote_code: dict[str, Any] | None


class HuggingFaceHubClient:
    def __init__(self, token: str | None = None) -> None:
        try:
            from huggingface_hub import HfApi
        except Exception as exc:  # pragma: no cover
            raise HFResolutionError(f"huggingface_hub import failed: {exc}")
        self._api = HfApi(token=token)

    def resolve_commit(self, repo_id: str, revision: str) -> str:
        info = self._api.model_info(repo_id=repo_id, revision=revision)
        sha = getattr(info, "sha", None)
        if not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{40}", sha):
            raise HFResolutionError(f"Unable to resolve immutable commit for {repo_id}@{revision}")
        return sha

    def list_files(self, repo_id: str, revision: str) -> list[str]:
        files = self._api.list_repo_files(repo_id=repo_id, revision=revision)
        if not isinstance(files, list):
            raise HFResolutionError(f"Unexpected list_repo_files response for {repo_id}@{revision}")
        return sorted(str(item) for item in files)

    def download_file(self, repo_id: str, revision: str, file_path: str, cache_dir: Path | None) -> Path:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:  # pragma: no cover
            raise HFResolutionError(f"huggingface_hub import failed: {exc}")

        target = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        return Path(target)


_MODEL_ROLE_TO_ARTIFACT_TYPE = {
    "weights_shard": "model_weights",
    "config": "model_config",
    "tokenizer": "tokenizer",
    "generation_config": "generation_config",
    "chat_template": "chat_template",
    "prompt_formatter": "prompt_formatter",
}


def parse_hf_source(source: str) -> str:
    match = re.fullmatch(r"hf://([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)", source)
    if not match:
        raise HFResolutionError(f"Invalid HF source: {source}")
    return match.group(1)


def _choose_first(candidates: list[str], files: set[str]) -> str | None:
    for item in candidates:
        if item in files:
            return item
    return None


def _select_required_paths(files: list[str]) -> dict[str, list[str] | str]:
    file_set = set(files)

    weights = sorted(
        [
            path
            for path in files
            if (
                path.endswith(".safetensors")
                or path.endswith(".bin")
            )
            and not path.endswith(".index.json")
            and (
                "model" in Path(path).name
                or "pytorch_model" in Path(path).name
                or Path(path).name.endswith(".safetensors")
            )
        ]
    )

    config = _choose_first(["config.json"], file_set)
    tokenizer = _choose_first(["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "tokenization_config.json"], file_set)
    generation_config = _choose_first(["generation_config.json"], file_set)
    chat_template = _choose_first(["chat_template.jinja", "tokenizer_config.json", "tokenization_config.json"], file_set)
    prompt_formatter = _choose_first(
        [
            "prompt_formatter.py",
            "prompt_formatting.py",
            "conversation.py",
            "chat_template.py",
            "tokenizer_config.json",
            "tokenization_config.json",
        ],
        file_set,
    )

    missing: list[str] = []
    if len(weights) == 0:
        missing.append("weights_shard")
    if config is None:
        missing.append("config")
    if tokenizer is None:
        missing.append("tokenizer")
    if generation_config is None:
        missing.append("generation_config")
    if chat_template is None:
        missing.append("chat_template")
    if prompt_formatter is None:
        missing.append("prompt_formatter")

    if missing:
        raise HFResolutionError("Required model files missing from HF repo: " + ", ".join(missing))

    return {
        "weights_shard": weights,
        "config": config,
        "tokenizer": tokenizer,
        "generation_config": generation_config,
        "chat_template": chat_template,
        "prompt_formatter": prompt_formatter,
    }


def _remote_code_digest(client: HFClient, repo_id: str, revision: str, python_files: list[str], cache_dir: Path | None) -> str:
    h = hashlib.sha256()
    for file_path in sorted(python_files):
        h.update(file_path.encode("utf-8"))
        h.update(b"\0")
        local = client.download_file(repo_id, revision, file_path, cache_dir)
        h.update(local.read_bytes())
        h.update(b"\0")
    return "sha256:" + h.hexdigest()


def _artifact_id(role: str, path: str) -> str:
    name = Path(path).name
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "-", name)
    return f"hf-{role}-{safe_name}"


def resolve_hf_model(
    model: dict[str, Any],
    trust_remote_code: bool,
    *,
    client: HFClient,
    cache_dir: Path | None = None,
) -> ResolvedHF:
    repo_id = parse_hf_source(model["source"])
    requested_revision = (
        model.get("requested_revision")
        or model.get("resolved_revision")
        or "main"
    )
    if not isinstance(requested_revision, str) or requested_revision.strip() == "":
        raise HFResolutionError("Model requested/resolved revision must be non-empty")

    commit = client.resolve_commit(repo_id, requested_revision)
    files = client.list_files(repo_id, commit)
    selected = _select_required_paths(files)

    required_files: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []

    def add_file(role: str, file_path: str) -> None:
        local = client.download_file(repo_id, commit, file_path, cache_dir)
        digest = sha256_file(local)
        size_bytes = local.stat().st_size
        uri = f"hf://{repo_id}/{file_path}"

        required_files.append(
            {
                "role": role,
                "path": file_path,
                "uri": uri,
                "digest": digest,
                "size_bytes": size_bytes,
            }
        )

        artifacts.append(
            {
                "artifact_id": _artifact_id(role, file_path),
                "artifact_type": _MODEL_ROLE_TO_ARTIFACT_TYPE[role],
                "name": Path(file_path).name,
                "source_kind": "hf",
                "source_uri": uri,
                "immutable_ref": commit,
                "expected_digest": digest,
                "size_bytes": size_bytes,
            }
        )

    for weights_path in selected["weights_shard"]:  # type: ignore[index]
        add_file("weights_shard", str(weights_path))

    for role in ["config", "tokenizer", "generation_config", "chat_template", "prompt_formatter"]:
        add_file(role, str(selected[role]))  # type: ignore[index]

    remote_code: dict[str, Any] | None = None
    if trust_remote_code:
        py_files = [item for item in files if item.endswith(".py")]
        if not py_files:
            raise HFResolutionError("trust_remote_code=true but no python files found in repository")
        rc_digest = _remote_code_digest(client, repo_id, commit, py_files, cache_dir)
        remote_code = {
            "commit": commit,
            "uri": f"hf://{repo_id}?revision={commit}#remote_code",
            "digest": rc_digest,
        }
        artifacts.append(
            {
                "artifact_id": "hf-remote-code",
                "artifact_type": "remote_code",
                "name": "remote_code",
                "source_kind": "hf",
                "source_uri": f"hf://{repo_id}?revision={commit}#remote_code",
                "immutable_ref": commit,
                "expected_digest": rc_digest,
                "size_bytes": 1,
            }
        )

    required_files = sorted(required_files, key=lambda item: (item["role"], item["path"]))
    artifacts = sorted(artifacts, key=lambda item: (item["artifact_type"], item["artifact_id"], item["expected_digest"]))

    return ResolvedHF(
        resolved_revision=commit,
        required_files=required_files,
        model_artifacts=artifacts,
        remote_code=remote_code,
    )
