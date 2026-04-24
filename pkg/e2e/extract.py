"""Extract prompt/output token IDs from a vLLM OpenAI-compatible chat response.

vLLM ≥0.10.2 returns `token_ids` on `choices[0]` for generated tokens and
`prompt_token_ids` (top level, or on `choices[0]`) for the tokenized
prompt when the request sets `return_token_ids: true`. These helpers
check both shapes and raise loudly when the IDs are unavailable — for
commitment use cases we must NOT silently fall back to pseudo-tokens,
because the resulting commitments would not reproduce on replay.
"""
from __future__ import annotations

from typing import Any


class TokenIdExtractionError(RuntimeError):
    """Raised when a response does not carry the expected token IDs."""


def _coerce_int_list(value: Any) -> list[int] | None:
    if isinstance(value, list):
        return [int(t) for t in value]
    return None


def extract_output_token_ids(response: dict[str, Any]) -> list[int]:
    """Return the list of output token IDs from a chat-completion response.

    Raises TokenIdExtractionError if the shape doesn't carry them — the
    caller almost certainly forgot to set `return_token_ids: true`.
    """
    top = _coerce_int_list(response.get("token_ids"))
    if top is not None:
        return top

    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        nested = _coerce_int_list(choices[0].get("token_ids"))
        if nested is not None:
            return nested

    raise TokenIdExtractionError(
        "Chat completion response is missing output token_ids. "
        "Set `return_token_ids: true` on the vLLM request (requires vLLM ≥0.10.2)."
    )


def extract_input_token_ids(response: dict[str, Any]) -> list[int]:
    """Return the list of prompt (input) token IDs from a chat response.

    vLLM surfaces these as `prompt_token_ids`, either at the top level or
    inside `choices[0]`, when `return_token_ids: true` is set on the
    request. Raises TokenIdExtractionError if absent — an empty prompt
    is still a list, so "missing" and "empty" are distinguished.
    """
    top = _coerce_int_list(response.get("prompt_token_ids"))
    if top is not None:
        return top

    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        nested = _coerce_int_list(choices[0].get("prompt_token_ids"))
        if nested is not None:
            return nested

    raise TokenIdExtractionError(
        "Chat completion response is missing prompt_token_ids. "
        "Set `return_token_ids: true` on the vLLM request (requires vLLM ≥0.10.2)."
    )
