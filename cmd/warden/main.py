#!/usr/bin/env python3
"""Inline active warden -- NFQUEUE-based packet normalizer.

Intercepts packets via iptables NFQUEUE, passes them through the
ActiveWarden MRF normalizer, and re-injects the scrubbed packets.

Requires root (for netfilter queue access) and the ``netfilterqueue``
Python package.

Usage:
    sudo python3 cmd/warden/main.py [--config /etc/warden/warden.yaml]

Or as a module (from the repo root):
    sudo python3 -m pkg.networkdet.warden_service_main [--config ...]
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

# Ensure the repo root is on sys.path so pkg.* imports work
# even when invoked directly as a script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pkg.networkdet.warden_config import load_config
from pkg.networkdet.warden_service import WardenService, _shutdown


logger = logging.getLogger("warden")


def _handle_signal(signum, frame):
    """Signal handler for graceful shutdown."""
    logger.info("Received signal %d, shutting down...", signum)
    _shutdown.set()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inline active warden (NFQUEUE packet normalizer)"
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to YAML config file (default: /etc/warden/warden.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    service = WardenService(config)
    service.run()


if __name__ == "__main__":
    main()
