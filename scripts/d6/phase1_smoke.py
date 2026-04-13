"""D6 Phase 1 smoke test — minimal vLLM inference for the determinism repeat.

Runs Qwen3-0.6B on a single GPU with greedy decoding and prints the int token
IDs. Two consecutive runs must produce identical TOKEN_IDS lines for Phase 1
to pass (see docs/plans/d6-lambda-staged-rollout.md Task 1.6).
"""
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    seed=42,
    enforce_eager=True,
    max_model_len=512,
    dtype="auto",
    attention_backend="FLASH_ATTN",
)
params = SamplingParams(temperature=0, max_tokens=20)
out = llm.generate(["The meaning of life is"], params)
gen = out[0].outputs[0]
print("TOKEN_IDS:", list(gen.token_ids))
print("TEXT:", gen.text)
