"""Verify that no pseudo image has entered the real training distribution."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    root = Path(args.dataset)
    images = sorted((root / "image").glob("*.png"))
    masks = sorted((root / "mask").glob("*.png"))
    if len(images) != 1 or len(masks) != 1:
        raise RuntimeError(
            f"Safety check failed: expected exactly 1 real pair, got "
            f"{len(images)} images and {len(masks)} masks."
        )
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("real_image_count") != 1:
        raise RuntimeError("metadata does not declare exactly one real image.")
    if metadata.get("fixed_mask_bank") is not False:
        raise RuntimeError("Dataset metadata does not disable the fixed mask bank.")
    print(
        "Validated: exactly 1 real image/mask pair; online sampling enabled; "
        "no fixed mask bank is required."
    )


if __name__ == "__main__":
    main()
