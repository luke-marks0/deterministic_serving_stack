from pose.memmap import MemoryMap


def test_total_blocks():
    mm = MemoryMap(dram_blocks=100, hbm_blocks=50, nvme_blocks=200)
    assert mm.total_blocks == 350


def test_dram_region():
    mm = MemoryMap(dram_blocks=100, hbm_blocks=50, nvme_blocks=200)
    region, local_idx = mm.resolve(0)
    assert region == "dram"
    assert local_idx == 0

    region, local_idx = mm.resolve(99)
    assert region == "dram"
    assert local_idx == 99


def test_hbm_region():
    mm = MemoryMap(dram_blocks=100, hbm_blocks=50, nvme_blocks=200)
    region, local_idx = mm.resolve(100)
    assert region == "hbm"
    assert local_idx == 0

    region, local_idx = mm.resolve(149)
    assert region == "hbm"
    assert local_idx == 49


def test_nvme_region():
    mm = MemoryMap(dram_blocks=100, hbm_blocks=50, nvme_blocks=200)
    region, local_idx = mm.resolve(150)
    assert region == "nvme"
    assert local_idx == 0

    region, local_idx = mm.resolve(349)
    assert region == "nvme"
    assert local_idx == 199


def test_out_of_range():
    mm = MemoryMap(dram_blocks=100, hbm_blocks=50, nvme_blocks=200)
    import pytest
    with pytest.raises(IndexError):
        mm.resolve(350)
