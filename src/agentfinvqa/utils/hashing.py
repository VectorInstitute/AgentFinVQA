"""SHA-256 hashing utilities."""

import hashlib


def sha256_file(path: str) -> str:
    """Compute SHA-256 hash of a file at the given path and return as hex string."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hash of given bytes data and return as hex string."""
    return hashlib.sha256(data).hexdigest()
