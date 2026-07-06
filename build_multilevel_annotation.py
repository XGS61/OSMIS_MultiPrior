"""Merge one-case binary annotations into an indexed generation condition."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_binary(path):
    return np.asarray(Image.open(path).convert("L")) >= 128


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", required=True, help="current primary/outer mask")
    parser.add_argument("--inner", required=True, help="internal hiatus mask")
    parser.add_argument("--output", required=True, help="indexed PNG output")
    args = parser.parse_args()

    outer = load_binary(args.outer)
    inner = load_binary(args.inner)
    if outer.shape != inner.shape:
        raise ValueError(f"Mask size mismatch: {outer.shape} vs {inner.shape}")
    outside_pixels = int((inner & ~outer).sum())
    if outside_pixels:
        raise ValueError(
            f"Internal mask has {outside_pixels} pixels outside the outer mask. "
            "Correct the annotation before training."
        )
    labels = np.zeros(outer.shape, dtype=np.uint8)
    labels[outer] = 1
    labels[inner] = 2
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(labels, mode="L").save(output)
    print(
        f"Saved indexed annotation to {output}: "
        f"outer-only={(labels == 1).sum()}, inner={(labels == 2).sum()}"
    )


if __name__ == "__main__":
    main()
