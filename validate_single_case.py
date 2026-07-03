"""Verify that no pseudo image has entered the real training distribution."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-priors", type=int, default=64)
    args = parser.parse_args()
    root = Path(args.dataset)
    images = sorted((root / "image").glob("*.png"))
    masks = sorted((root / "mask").glob("*.png"))
    priors = sorted((root / "mask_priors").glob("*.png"))
    if len(images) != 1 or len(masks) != 1:
        raise RuntimeError(
            f"Safety check failed: expected exactly 1 real pair, got "
            f"{len(images)} images and {len(masks)} masks."
        )
    if len(priors) != args.expected_priors:
        raise RuntimeError(
            f"Expected {args.expected_priors} mask priors, found {len(priors)}."
        )
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("real_image_count") != 1:
        raise RuntimeError("metadata does not declare exactly one real image.")
    print(f"Validated: 1 real image, 1 real mask, {len(priors)} mask-only priors.")


if __name__ == "__main__":
    main()
