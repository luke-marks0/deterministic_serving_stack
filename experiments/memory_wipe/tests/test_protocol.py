from pose.protocol import run_protocol
from pose.memory.dram import DramRegion

BLOCK_SIZE = 4096


def test_protocol_passes_with_honest_prover():
    """Honest prover passes all challenges."""
    dram = DramRegion(size_bytes=100 * BLOCK_SIZE, block_size=BLOCK_SIZE)
    result = run_protocol(
        regions={"dram": dram},
        block_size=BLOCK_SIZE,
        num_rounds=50,
    )
    assert result.passed is True
    assert result.rounds_passed == 50
    assert result.rounds_total == 50


def test_protocol_reports_coverage():
    """Protocol reports how many bytes were wiped and coverage ratio."""
    dram = DramRegion(size_bytes=100 * BLOCK_SIZE, block_size=BLOCK_SIZE)
    result = run_protocol(
        regions={"dram": dram},
        region_info={"dram": {
            "total_bytes": 110 * BLOCK_SIZE,
            "reserved_bytes": 10 * BLOCK_SIZE,
            "reserved_reason": "test reserve",
        }},
        block_size=BLOCK_SIZE,
        num_rounds=10,
    )
    assert result.bytes_wiped == 100 * BLOCK_SIZE
    assert result.bytes_total == 110 * BLOCK_SIZE
    assert result.bytes_reserved == 10 * BLOCK_SIZE
    assert 0.90 < result.coverage < 0.92  # ~90.9%


def test_protocol_has_per_region_metrics():
    """Each region reports fill time and throughput."""
    dram = DramRegion(size_bytes=100 * BLOCK_SIZE, block_size=BLOCK_SIZE)
    result = run_protocol(
        regions={"dram": dram},
        block_size=BLOCK_SIZE,
        num_rounds=10,
    )
    assert len(result.region_metrics) == 1
    assert result.region_metrics[0].name == "dram"
    assert result.region_metrics[0].fill_time_s > 0
    assert result.region_metrics[0].fill_throughput_gbps > 0


def test_protocol_measures_resume_time():
    """Protocol measures time to resume after wipe."""
    dram = DramRegion(size_bytes=100 * BLOCK_SIZE, block_size=BLOCK_SIZE)
    result = run_protocol(
        regions={"dram": dram},
        block_size=BLOCK_SIZE,
        num_rounds=10,
    )
    assert result.resume_time_s >= 0
