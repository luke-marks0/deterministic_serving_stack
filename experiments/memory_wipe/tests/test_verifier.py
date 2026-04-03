import os
from pose.verifier import Verifier
from pose.noise import generate_block


def test_verifier_seed_is_random():
    v1 = Verifier(total_blocks=1000, block_size=4096)
    v2 = Verifier(total_blocks=1000, block_size=4096)
    assert v1.seed != v2.seed


def test_noise_stream_matches_individual_blocks():
    """The stream the prover receives matches what the verifier checks against."""
    v = Verifier(total_blocks=100, block_size=4096)
    stream = list(v.noise_stream())
    assert len(stream) == 100
    for i, block in enumerate(stream):
        assert block == generate_block(v.seed, i, 4096)


def test_noise_stream_does_not_expose_seed():
    """The stream yields bytes, not the seed itself."""
    v = Verifier(total_blocks=10, block_size=4096)
    for block in v.noise_stream():
        assert isinstance(block, bytes)
        assert len(block) == 4096
        assert block != v.seed  # Not the seed


def test_challenge_in_range():
    v = Verifier(total_blocks=1000, block_size=4096)
    for _ in range(100):
        idx = v.challenge()
        assert 0 <= idx < 1000


def test_verify_correct():
    v = Verifier(total_blocks=100, block_size=4096)
    stream = list(v.noise_stream())
    idx = v.challenge()
    assert v.verify(idx, stream[idx]) is True


def test_verify_incorrect():
    v = Verifier(total_blocks=1000, block_size=4096)
    idx = v.challenge()
    assert v.verify(idx, b"\x00" * 4096) is False
