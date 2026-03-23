"""Configuration loader for the inline active warden service.

Reads from a YAML config file or falls back to environment variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Optional YAML dependency -- fall back to env vars if not installed.
try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


_DEFAULT_CONFIG_PATH = "/etc/warden/warden.yaml"


@dataclass
class WardenConfig:
    """Configuration for the inline warden service."""

    # Secret key used for deterministic IP-ID encryption and ISN rewriting.
    secret: bytes = b"warden-default-key"

    # Normalized TTL value injected into every IP packet.
    ttl: int = 64

    # NFQUEUE number to bind to (must match iptables rule).
    queue_num: int = 0

    # Maximum number of packets held in the kernel queue at once.
    max_queue_len: int = 4096

    # How often (in seconds) to log stats.
    stats_interval: float = 30.0

    # Log level (DEBUG, INFO, WARNING, ERROR).
    log_level: str = "INFO"

    # iptables chain to hook (FORWARD for bridge/gateway, INPUT/OUTPUT for host).
    chain: str = "FORWARD"


def load_config(path: str | None = None) -> WardenConfig:
    """Load configuration, preferring YAML file then environment variables.

    Priority (highest first):
      1. Environment variables (WARDEN_SECRET, WARDEN_TTL, etc.)
      2. YAML config file
      3. Defaults
    """
    cfg = WardenConfig()

    # --- Load from YAML if available ---
    config_path = path or os.environ.get("WARDEN_CONFIG", _DEFAULT_CONFIG_PATH)
    if _HAS_YAML and Path(config_path).is_file():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        if "secret" in data:
            cfg.secret = data["secret"].encode() if isinstance(data["secret"], str) else data["secret"]
        if "ttl" in data:
            cfg.ttl = int(data["ttl"])
        if "queue_num" in data:
            cfg.queue_num = int(data["queue_num"])
        if "max_queue_len" in data:
            cfg.max_queue_len = int(data["max_queue_len"])
        if "stats_interval" in data:
            cfg.stats_interval = float(data["stats_interval"])
        if "log_level" in data:
            cfg.log_level = str(data["log_level"]).upper()
        if "chain" in data:
            cfg.chain = str(data["chain"]).upper()

    # --- Override from environment variables ---
    if "WARDEN_SECRET" in os.environ:
        cfg.secret = os.environ["WARDEN_SECRET"].encode()
    if "WARDEN_TTL" in os.environ:
        cfg.ttl = int(os.environ["WARDEN_TTL"])
    if "WARDEN_QUEUE_NUM" in os.environ:
        cfg.queue_num = int(os.environ["WARDEN_QUEUE_NUM"])
    if "WARDEN_MAX_QUEUE_LEN" in os.environ:
        cfg.max_queue_len = int(os.environ["WARDEN_MAX_QUEUE_LEN"])
    if "WARDEN_STATS_INTERVAL" in os.environ:
        cfg.stats_interval = float(os.environ["WARDEN_STATS_INTERVAL"])
    if "WARDEN_LOG_LEVEL" in os.environ:
        cfg.log_level = os.environ["WARDEN_LOG_LEVEL"].upper()
    if "WARDEN_CHAIN" in os.environ:
        cfg.chain = os.environ["WARDEN_CHAIN"].upper()

    return cfg
