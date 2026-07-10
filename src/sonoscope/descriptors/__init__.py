"""Descriptor derivation: frozen interpretation thresholds + hashing (by design).

This package turns deterministic features into human-readable descriptors. Cycle 1
establishes the frozen, hashed threshold set (``thresholds.py``) that versions the
interpretation layer independently of the feature-extraction ``params_sha256``.
"""
