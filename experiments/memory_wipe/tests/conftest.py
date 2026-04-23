import shutil
import pytest

def cuda_available():
    try:
        from pose.detect import get_cuda_runtime
        return get_cuda_runtime().device_count() > 0
    except Exception:
        return False

def nvme_device_available():
    """Check if a test NVMe path exists."""
    import os
    return os.path.exists(os.environ.get("POSE_NVME_PATH", ""))

# Auto-skip markers
def pytest_collection_modifyitems(items):
    for item in items:
        if "cuda" in item.keywords and not cuda_available():
            item.add_marker(pytest.mark.skip(reason="No CUDA GPU"))
        if "nvme" in item.keywords and not nvme_device_available():
            item.add_marker(pytest.mark.skip(reason="No NVMe device"))
