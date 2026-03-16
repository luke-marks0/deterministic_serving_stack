"""Large batch determinism tests.

Verifies that determinism holds at production-scale batch sizes (128+).
vLLM's default max_num_seqs is 256 — these tests push toward that
limit to catch nondeterminism that only appears under heavy batching.

Three test tiers:
1. 128 concurrent requests, compare outputs across two runs
2. 128 concurrent requests with varied prompt lengths (ragged batches)
3. 128 concurrent requests, compare egress digests across two runs

Requires a running vLLM server. Skipped in CI (no GPU).

Environment:
    DETERMINISTIC_SERVER_URL: server URL (default: http://127.0.0.1:8000)
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import random
import unittest
from urllib.request import Request, urlopen

SERVER_URL = os.getenv("DETERMINISTIC_SERVER_URL", "http://127.0.0.1:8000")
BATCH_SIZE = int(os.getenv("DETERMINISTIC_BATCH_SIZE", "128"))


def _server_available() -> bool:
    try:
        urlopen(f"{SERVER_URL}/health", timeout=3)
        return True
    except Exception:
        return False


def _chat(prompt: str, max_tokens: int = 32, seed: int = 42) -> dict:
    import time
    body = json.dumps({
        "model": "Qwen/Qwen3-1.7B",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": seed,
    }).encode()
    for attempt in range(3):
        try:
            req = Request(f"{SERVER_URL}/v1/chat/completions", data=body,
                          headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=300) as resp:
                return json.loads(resp.read())
        except (ConnectionResetError, OSError) as e:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def _send_batch(prompts: list[str], max_tokens: int = 32) -> list[str]:
    """Send all prompts concurrently and return content strings in order."""
    results: dict[int, str] = {}

    def _do(idx: int, prompt: str) -> tuple[int, str]:
        r = _chat(prompt, max_tokens=max_tokens)
        return idx, r["choices"][0]["message"]["content"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(prompts), 32)) as pool:
        futs = [pool.submit(_do, i, p) for i, p in enumerate(prompts)]
        for fut in concurrent.futures.as_completed(futs):
            idx, content = fut.result()
            results[idx] = content

    return [results[i] for i in range(len(prompts))]


def _egress_digest(contents: list[str]) -> str:
    h = hashlib.sha256()
    for c in contents:
        h.update(hashlib.sha256(c.encode("utf-8")).digest())
    return f"sha256:{h.hexdigest()}"


def _make_uniform_prompts(n: int) -> list[str]:
    """N prompts of similar length."""
    topics = [
        "quantum computing", "photosynthesis", "general relativity",
        "machine learning", "plate tectonics", "the water cycle",
        "DNA replication", "neural networks", "black holes",
        "the Fibonacci sequence", "encryption", "compilers",
        "the speed of light", "vaccines", "nuclear fusion",
        "CRISPR gene editing", "the Standard Model", "blockchain",
    ]
    return [f"Explain {topics[i % len(topics)]} in one sentence. (variant {i})"
            for i in range(n)]


def _make_ragged_prompts(n: int) -> list[str]:
    """N prompts with intentionally varied lengths to stress chunked prefill."""
    rng = random.Random(42)  # deterministic
    prompts = []
    for i in range(n):
        length_class = rng.choice(["short", "medium", "long"])
        if length_class == "short":
            prompts.append(f"What is {i}+{i}?")
        elif length_class == "medium":
            padding = "This requires careful thought. " * rng.randint(5, 15)
            prompts.append(f"{padding}Now answer: what is the capital of country number {i}?")
        else:
            padding = "Consider the following detailed context. " * rng.randint(20, 50)
            prompts.append(f"{padding}Given all of the above, summarize in one sentence. (variant {i})")
    return prompts


@unittest.skipUnless(_server_available(), "No server at " + SERVER_URL)
class TestLargeBatchDeterminism(unittest.TestCase):
    """128 concurrent uniform requests — outputs must match across runs."""

    def test_128_uniform_requests_deterministic(self):
        prompts = _make_uniform_prompts(BATCH_SIZE)

        print(f"\n  Run 1: {BATCH_SIZE} concurrent requests...")
        results1 = _send_batch(prompts, max_tokens=32)

        print(f"  Run 2: {BATCH_SIZE} concurrent requests...")
        results2 = _send_batch(prompts, max_tokens=32)

        mismatches = []
        for i, (a, b) in enumerate(zip(results1, results2)):
            if a != b:
                mismatches.append(i)

        if mismatches:
            for i in mismatches[:5]:
                print(f"    [{i}] {prompts[i][:50]}...")
                print(f"      run1: {results1[i][:60]}...")
                print(f"      run2: {results2[i][:60]}...")

        self.assertEqual(len(mismatches), 0,
                         f"{len(mismatches)}/{BATCH_SIZE} requests differ between runs")

    def test_128_uniform_egress_digest_matches(self):
        prompts = _make_uniform_prompts(BATCH_SIZE)

        results1 = _send_batch(prompts, max_tokens=32)
        results2 = _send_batch(prompts, max_tokens=32)

        d1 = _egress_digest(results1)
        d2 = _egress_digest(results2)

        self.assertEqual(d1, d2,
                         f"Egress digest mismatch at batch size {BATCH_SIZE}")


@unittest.skipUnless(_server_available(), "No server at " + SERVER_URL)
class TestRaggedBatchDeterminism(unittest.TestCase):
    """128 concurrent requests with varied lengths — stress chunked prefill."""

    def test_128_ragged_requests_deterministic(self):
        prompts = _make_ragged_prompts(BATCH_SIZE)

        # Show length distribution
        lengths = [len(p) for p in prompts]
        print(f"\n  Prompt lengths: min={min(lengths)} max={max(lengths)} "
              f"mean={sum(lengths)//len(lengths)}")

        print(f"  Run 1: {BATCH_SIZE} ragged requests...")
        results1 = _send_batch(prompts, max_tokens=64)

        print(f"  Run 2: {BATCH_SIZE} ragged requests...")
        results2 = _send_batch(prompts, max_tokens=64)

        mismatches = []
        for i, (a, b) in enumerate(zip(results1, results2)):
            if a != b:
                mismatches.append(i)

        if mismatches:
            for i in mismatches[:5]:
                print(f"    [{i}] len={len(prompts[i])} {prompts[i][:40]}...")
                print(f"      run1: {results1[i][:60]}...")
                print(f"      run2: {results2[i][:60]}...")

        self.assertEqual(len(mismatches), 0,
                         f"{len(mismatches)}/{BATCH_SIZE} ragged requests differ")

    def test_128_ragged_egress_digest_matches(self):
        prompts = _make_ragged_prompts(BATCH_SIZE)

        results1 = _send_batch(prompts, max_tokens=64)
        results2 = _send_batch(prompts, max_tokens=64)

        d1 = _egress_digest(results1)
        d2 = _egress_digest(results2)

        self.assertEqual(d1, d2,
                         f"Ragged batch egress digest mismatch at {BATCH_SIZE}")


if __name__ == "__main__":
    unittest.main()
