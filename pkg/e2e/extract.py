"""Extract the output token IDs from a vLLM OpenAI-compatible chat response.

vLLM ≥0.10.2 returns `token_ids` on `choices[0]` when the request sets
`return_token_ids: true`. Some versions also surface the list at the top
level. This helper checks both shapes and raises loudly when token IDs
are unavailable — for commitment use cases we must NOT silently fall
back to pseudo-tokens, because the resulting commitments would not
reproduce on replay.
"""
from __future__ import annotations

from typing import Any


class TokenIdExtractionError(RuntimeError):
    """Raised when a response does not carry the generated token IDs."""


def extract_output_token_ids(response: dict[str, Any]) -> list[int]:
    """Return the list of output token IDs from a chat-completion response.

    Raises TokenIdExtractionError if the shape doesn't carry them — the
    caller almost certainly forgot to set `return_token_ids: true`.
    """
    top = response.get("token_ids")
    if isinstance(top, list):
        return [int(t) for t in top]

    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        tids = choices[0].get("token_ids")
        if isinstance(tids, list):
            return [int(t) for t in tids]

    raise TokenIdExtractionError(
        "Chat completion response is missing output token_ids. "
        "Set `return_token_ids: true` on the vLLM request (requires vLLM ≥0.10.2)."
    )
