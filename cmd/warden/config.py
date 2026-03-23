"""Configuration for the warden service.

This is a convenience re-export; the actual implementation lives in
pkg/networkdet/warden_config.py to avoid the stdlib 'cmd' module
name collision.
"""
# Re-export for documentation purposes. The actual entry point
# (cmd/warden/main.py) imports from pkg.networkdet.warden_config directly.
