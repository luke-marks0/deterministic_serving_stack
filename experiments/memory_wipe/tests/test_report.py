import json
from pose.protocol import ProtocolResult, RegionMetrics
from pose.report import generate_report


def _sample_result():
    return ProtocolResult(
        passed=True,
        rounds_passed=100,
        rounds_total=100,
        bytes_wiped=1000 * 4096,
        bytes_total=1100 * 4096,
        bytes_reserved=100 * 4096,
        coverage=1000 / 1100,
        fill_time_s=2.5,
        verify_time_s=0.05,
        resume_time_s=0.3,
        region_metrics=[
            RegionMetrics(
                name="dram", total_bytes=500 * 4096, reserved_bytes=50 * 4096,
                reserved_reason="OS kernel", wiped_bytes=450 * 4096,
                fill_time_s=1.0, fill_throughput_gbps=1.8,
            ),
            RegionMetrics(
                name="hbm", total_bytes=300 * 4096, reserved_bytes=30 * 4096,
                reserved_reason="CUDA driver", wiped_bytes=270 * 4096,
                fill_time_s=0.5, fill_throughput_gbps=2.1,
            ),
            RegionMetrics(
                name="nvme", total_bytes=300 * 4096, reserved_bytes=20 * 4096,
                reserved_reason="FS metadata", wiped_bytes=280 * 4096,
                fill_time_s=1.0, fill_throughput_gbps=1.1,
            ),
        ],
        seed=b"\x00" * 32,
    )


def test_report_has_required_keys():
    report = generate_report(_sample_result())
    assert "wipe_time_s" in report
    assert "resume_time_s" in report
    assert "memory_inventory" in report
    assert "coverage_pct" in report
    assert "verification" in report


def test_report_is_json_serializable():
    report = generate_report(_sample_result())
    dumped = json.dumps(report)
    loaded = json.loads(dumped)
    assert loaded["wipe_time_s"] == 2.5


def test_report_regions_have_reserved_reason():
    report = generate_report(_sample_result())
    for region in report["memory_inventory"]["regions"]:
        assert "reserved_bytes" in region
        assert "reserved_reason" in region
        assert len(region["reserved_reason"]) > 0
