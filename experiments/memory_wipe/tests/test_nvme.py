import os
import tempfile
import pytest
from pose.memory.nvme import NvmeRegion

BLOCK_SIZE = 4096
pytestmark = pytest.mark.nvme


@pytest.fixture
def nvme_dir():
    """Use POSE_NVME_PATH env var, or skip."""
    path = os.environ.get("POSE_NVME_PATH")
    if not path or not os.path.isdir(path):
        pytest.skip("Set POSE_NVME_PATH to an NVMe-mounted directory")
    return path


def test_write_read_roundtrip(nvme_dir):
    filepath = os.path.join(nvme_dir, "test_region.bin")
    region = NvmeRegion(filepath, num_blocks=10, block_size=BLOCK_SIZE)
    data = os.urandom(BLOCK_SIZE)
    region.write_block(0, data)
    assert region.read_block(0) == data
    region.close()
    os.unlink(filepath)


def test_multiple_blocks(nvme_dir):
    filepath = os.path.join(nvme_dir, "test_multi.bin")
    region = NvmeRegion(filepath, num_blocks=10, block_size=BLOCK_SIZE)
    blocks = [os.urandom(BLOCK_SIZE) for _ in range(10)]
    for i, b in enumerate(blocks):
        region.write_block(i, b)
    for i, b in enumerate(blocks):
        assert region.read_block(i) == b
    region.close()
    os.unlink(filepath)


def test_index_out_of_range(nvme_dir):
    filepath = os.path.join(nvme_dir, "test_oor.bin")
    region = NvmeRegion(filepath, num_blocks=10, block_size=BLOCK_SIZE)
    with pytest.raises(IndexError):
        region.read_block(10)
    region.close()
    os.unlink(filepath)
