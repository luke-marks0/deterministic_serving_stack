"""NIC and network configuration conformance validation.

Validates observed NIC hardware against the manifest hardware profile
to ensure deterministic networking constraints are met.
"""
from __future__ import annotations

from typing import Any


def validate_nic_config(
    manifest: dict[str, Any],
    observed_nic: dict[str, Any],
) -> list[str]:
    """Validate observed NIC config against the manifest.

    Returns a list of conformance violation messages.
    An empty list means the NIC is conformant.
    """
    violations: list[str] = []
    expected_nic = manifest.get("hardware_profile", {}).get("nic", {})

    checks = [
        ("model", "NIC model"),
        ("pci_id", "NIC PCI ID"),
        ("firmware", "NIC firmware version"),
    ]
    for field, label in checks:
        expected = expected_nic.get(field)
        actual = observed_nic.get(field)
        if expected is not None and actual is not None and expected != actual:
            violations.append(
                f"{label} mismatch: expected={expected!r}, actual={actual!r}"
            )

    expected_speed = expected_nic.get("link_speed_gbps")
    actual_speed = observed_nic.get("link_speed_gbps")
    if expected_speed is not None and actual_speed is not None:
        if expected_speed != actual_speed:
            violations.append(
                f"NIC link speed mismatch: expected={expected_speed}Gbps, actual={actual_speed}Gbps"
            )

    expected_offloads = expected_nic.get("offloads", {})
    actual_offloads = observed_nic.get("offloads", {})
    for offload_key in ("checksum", "tso", "gso", "vlan_strip"):
        expected_val = expected_offloads.get(offload_key)
        actual_val = actual_offloads.get(offload_key)
        if expected_val is not None and actual_val is not None and expected_val != actual_val:
            violations.append(
                f"NIC offload '{offload_key}' mismatch: expected={expected_val}, actual={actual_val}"
            )

    return violations
