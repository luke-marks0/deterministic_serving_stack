import os
import pytest
from pose.memory.hbm import HbmRegion

BLOCK_SIZE = 4096
pytestmark = pytest.mark.cuda


def test_capacity():
    region = HbmRegion(size_bytes=BLOCK_SIZE * 10, block_size=BLOCK_SIZE)
    assert region.num_blocks == 10
    region.close()


def test_write_read_roundtrip():
    region = HbmRegion(size_bytes=BLOCK_SIZE * 10, block_size=BLOCK_SIZE)
    data = os.urandom(BLOCK_SIZE)
    region.write_block(0, data)
    assert region.read_block(0) == data
    region.close()


def test_write_multiple_blocks():
    region = HbmRegion(size_bytes=BLOCK_SIZE * 10, block_size=BLOCK_SIZE)
    blocks = [os.urandom(BLOCK_SIZE) for _ in range(10)]
    for i, b in enumerate(blocks):
        region.write_block(i, b)
    for i, b in enumerate(blocks):
        assert region.read_block(i) == b
    region.close()


def test_index_out_of_range():
    region = HbmRegion(size_bytes=BLOCK_SIZE * 10, block_size=BLOCK_SIZE)
    with pytest.raises(IndexError):
        region.read_block(10)
    region.close()
