from pose.hwinfo import collect_hwinfo


def test_hwinfo_has_required_keys():
    info = collect_hwinfo()
    assert "hostname" in info
    assert "cpu_model" in info
    assert "arch" in info
    assert "gpu_name" in info


def test_hwinfo_values_are_strings():
    info = collect_hwinfo()
    assert isinstance(info["hostname"], str)
    assert isinstance(info["cpu_model"], str)
