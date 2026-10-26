"""Regenerate the parity fixtures in test/fixtures: the expected RGB bytes for
each fixed seed, rendered by the Python reference engine.
"""
from __future__ import annotations
import os
from latent import seed_to_latent, hex_to_be32
from engine import render_rgb
from export import load_qmodel

HERE = os.path.dirname(__file__)
FIX = os.path.join(HERE, "..", "test", "fixtures")

SEEDS = [("c0ffeebabe"),
         ("82a46b4e893c75b8dddab3ec26439e49f9197d79c9bb64001ddf12ebb685a98a"),
         ("cafef00d"), ("deadbeef"),
         ("ff" * 32), ("a5" * 32), ("0f" * 32), ("1234567890abcdef")]


def main():
    os.makedirs(FIX, exist_ok=True)
    m = load_qmodel()
    for i, seed in enumerate(SEEDS):
        rgb = render_rgb(m, seed_to_latent(hex_to_be32(seed)))
        open(os.path.join(FIX, f"expected_{i}.hex"), "w").write("0x" + rgb.hex())
    print(f"wrote {len(SEEDS)} expected fixtures -> {FIX}")


if __name__ == "__main__":
    main()
