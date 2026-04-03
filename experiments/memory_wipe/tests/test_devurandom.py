import os
import tempfile
from pose.devurandom import pregen_urandom, stream_from_file, verify_from_file

BLOCK_SIZE = 4096


def test_pregen_creates_file_of_correct_size():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "noise.bin")
        pregen_urandom(path, total_bytes=1024 * 1024, num_cores=2)
        assert os.path.getsize(path) == 1024 * 1024


def test_stream_from_file_yields_correct_chunks():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "noise.bin")
        total = 4 * BLOCK_SIZE
        pregen_urandom(path, total_bytes=total, num_cores=1)
        chunks = list(stream_from_file(path, offset=0, size=total, chunk_size=BLOCK_SIZE))
        assert len(chunks) == 4
        assert all(len(c) == BLOCK_SIZE for c in chunks)


def test_verify_from_file_correct():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "noise.bin")
        total = 10 * BLOCK_SIZE
        pregen_urandom(path, total_bytes=total, num_cores=1)
        # Read block 5 back from file, then verify it
        block = verify_from_file(path, block_index=5, block_size=BLOCK_SIZE)
        assert len(block) == BLOCK_SIZE
        # Read it again — should be identical (deterministic file)
        block2 = verify_from_file(path, block_index=5, block_size=BLOCK_SIZE)
        assert block == block2


def test_verify_from_file_different_blocks_differ():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "noise.bin")
        total = 10 * BLOCK_SIZE
        pregen_urandom(path, total_bytes=total, num_cores=1)
        b0 = verify_from_file(path, block_index=0, block_size=BLOCK_SIZE)
        b1 = verify_from_file(path, block_index=1, block_size=BLOCK_SIZE)
        assert b0 != b1
