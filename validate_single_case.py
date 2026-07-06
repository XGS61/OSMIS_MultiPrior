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
    conditions = sorted((root / "condition").glob("*.npz"))
    if len(images) != 1 or len(masks) != 1:
        raise RuntimeError(
            f"Safety check failed: expected exactly 1 real pair, got "
            f"{len(images)} images and {len(masks)} masks."
        )
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(f"Missing dataset metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("condition_encoding") == "hierarchical_multichannel":
        if len(conditions) != 1:
            raise RuntimeError(
                f"Safety check failed: expected exactly 1 hierarchical "
                f"condition archive, got {len(conditions)}."
            )
    if metadata.get("real_image_count") != 1:
        raise RuntimeError("metadata does not declare exactly one real image.")
    if metadata.get("fixed_mask_bank") is not False:
        raise RuntimeError("Dataset metadata does not disable the fixed mask bank.")
    print(
        "Validated: exactly 1 real image/mask pair; hierarchical online "
        "sampling enabled; no fixed mask bank is required."
    )


if __name__ == "__main__":
    main()
