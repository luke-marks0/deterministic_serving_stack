import os
from pose.noise import generate_block, generate_blocks

SEED = os.urandom(32)
BLOCK_SIZE = 4096


def test_deterministic():
    """Same seed + index always gives the same block."""
    a = generate_block(SEED, index=42, block_size=BLOCK_SIZE)
    b = generate_block(SEED, index=42, block_size=BLOCK_SIZE)
    assert a == b


def test_correct_size():
    block = generate_block(SEED, index=0, block_size=BLOCK_SIZE)
    assert len(block) == BLOCK_SIZE


def test_different_indices_differ():
    a = generate_block(SEED, index=0, block_size=BLOCK_SIZE)
    b = generate_block(SEED, index=1, block_size=BLOCK_SIZE)
    assert a != b


def test_different_seeds_differ():
    a = generate_block(b"\x00" * 32, index=0, block_size=BLOCK_SIZE)
    b = generate_block(b"\x01" * 32, index=0, block_size=BLOCK_SIZE)
    assert a != b


def test_generate_blocks_sequential():
    """Bulk generation matches individual generation."""
    blocks = list(generate_blocks(SEED, start=5, count=3, block_size=BLOCK_SIZE))
    assert len(blocks) == 3
    for i, block in enumerate(blocks):
        assert block == generate_block(SEED, index=5 + i, block_size=BLOCK_SIZE)


def test_not_all_zeros():
    """Output is not trivially zero (sanity check)."""
    block = generate_block(SEED, index=0, block_size=BLOCK_SIZE)
    assert block != b"\x00" * BLOCK_SIZE
