"""Commitment -> latent, byte-identical to the onchain keccak derivation.

For batch 0..3, hash = keccak256(seed32 || uint256(batch));
latent[batch*8 + j] = int8((hash >> (j*8)) & 0xFF), j = 0..7.
"""
from __future__ import annotations
from eth_hash.auto import keccak

Z_DIM = 32


def to_i8(b: int) -> int:
    return b - 256 if b >= 128 else b


def seed_to_latent(seed_be32: bytes) -> list:
    if len(seed_be32) != 32:
        raise ValueError("seed must be 32 bytes")
    latent = [0] * Z_DIM
    for batch in range(4):
        h = int.from_bytes(keccak(seed_be32 + batch.to_bytes(32, "big")), "big")
        for j in range(8):
            latent[batch * 8 + j] = to_i8((h >> (j * 8)) & 0xFF)
    return latent


def hex_to_be32(h: str) -> bytes:
    h = h.lower().replace("0x", "")
    if len(h) > 64:
        raise ValueError("seed hex too long")
    return bytes.fromhex(h.zfill(64))
