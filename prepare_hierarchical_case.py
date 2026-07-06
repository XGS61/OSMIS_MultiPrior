"""Prepare one real image with hierarchical and exclusive anatomy conditions."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


REQUIRED_KEYS = (
    "rendered_support",
    "levator_hiatus",
    "internal_anterior_candidate",
    "internal_middle_candidate",
    "internal_posterior_candidate",
    "hiatus_boundary",
    "hiatus_signed_distance",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--indexed-mask", required=True)
    parser.add_argument("--conditions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--crop-top", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"))
    indexed = np.asarray(Image.open(args.indexed_mask).convert("L"))
    archive = np.load(args.conditions)
    missing = [key for key in REQUIRED_KEYS if key not in archive]
    if missing:
        raise ValueError(f"Missing hierarchical condition channels: {missing}")
    channels = np.stack([archive[key] for key in REQUIRED_KEYS]).astype(np.float32)

    if image.shape[:2] != indexed.shape or image.shape[:2] != channels.shape[1:]:
        raise ValueError(
            f"Shape mismatch: image={image.shape[:2]}, indexed={indexed.shape}, "
            f"conditions={channels.shape[1:]}"
        )
    if args.crop_top:
        image = image[args.crop_top:]
        indexed = indexed[args.crop_top:]
        channels = channels[:, args.crop_top:]

    binary_channels = channels[:6] >= 0.5
    support, hiatus = binary_channels[0], binary_channels[1]
    internals = binary_channels[2:5]
    if np.any(hiatus & ~support):
        raise ValueError("The levator-hiatus channel must remain inside support.")
    if np.any(internals & ~hiatus[None]):
        raise ValueError("Every internal candidate must remain inside the hiatus.")
    if np.any(internals.sum(axis=0) > 1):
        raise ValueError("Internal candidate channels must not overlap.")

    output = Path(args.output)
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    for name in ("image", "mask", "condition"):
        (output / name).mkdir(parents=True, exist_ok=True)
    if any((output / "image").glob("*")):
        raise FileExistsError(f"{output} is not empty; pass --overwrite.")

    Image.fromarray(image).save(output / "image" / "00000.png")
    Image.fromarray(indexed.astype(np.uint8)).save(output / "mask" / "00000.png")
    np.savez_compressed(
        output / "condition" / "00000.npz",
        conditions=channels.astype(np.float32),
        channel_names=np.asarray(REQUIRED_KEYS),
    )
    metadata = {
        "source_image": str(Path(args.image).resolve()),
        "source_indexed_mask": str(Path(args.indexed_mask).resolve()),
        "source_conditions": str(Path(args.conditions).resolve()),
        "crop_top": args.crop_top,
        "real_image_count": 1,
        "real_mask_count": 1,
        "real_condition_count": 1,
        "condition_encoding": "hierarchical_multichannel",
        "condition_channels": list(REQUIRED_KEYS),
        "num_condition_channels": len(REQUIRED_KEYS),
        "num_style_regions": int(indexed.max()) + 1,
        "online_sampling": True,
        "fixed_mask_bank": False,
        "review_required": (
            "Internal candidates are position/intensity-derived review labels. "
            "They are not independently validated clinical ground truth."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Prepared one hierarchical real pair at {output}; "
        f"{len(REQUIRED_KEYS)} structural channels and "
        f"{int(indexed.max()) + 1} exclusive style regions."
    )


if __name__ == "__main__":
    main()
