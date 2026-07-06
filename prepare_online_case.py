"""Prepare exactly one real image and one binary or indexed anatomy label map."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def largest_component(binary):
    labels, count = ndimage.label(binary)
    if count == 0:
        raise ValueError("An annotated class is empty.")
    sizes = ndimage.sum(binary, labels, range(1, count + 1))
    return labels == int(np.argmax(sizes) + 1)


def clean_label_map(raw):
    values = np.unique(raw)
    if values.max() > 30:
        binary = ndimage.binary_fill_holes(largest_component(raw >= 128))
        return (binary.astype(np.uint8) * 255), 2, "binary"

    labels = raw.astype(np.uint8)
    cleaned = np.zeros_like(labels)
    for class_index in range(1, int(labels.max()) + 1):
        region = labels == class_index
        if not region.any():
            raise ValueError(
                f"Indexed label map skips required class {class_index}."
            )
        cleaned[ndimage.binary_fill_holes(largest_component(region))] = class_index
    return cleaned, int(cleaned.max()) + 1, "indexed_multiclass"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--crop-top", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"))
    raw_mask = np.asarray(Image.open(args.mask).convert("L"))
    if image.shape[:2] != raw_mask.shape:
        raise ValueError(
            f"Image/mask size mismatch: {image.shape[:2]} vs {raw_mask.shape}"
        )
    if args.crop_top:
        image = image[args.crop_top:]
        raw_mask = raw_mask[args.crop_top:]
    mask, classes, encoding = clean_label_map(raw_mask)

    output = Path(args.output)
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    for name in ("image", "mask"):
        (output / name).mkdir(parents=True, exist_ok=True)
    if any((output / "image").glob("*")):
        raise FileExistsError(f"{output} is not empty; pass --overwrite.")

    Image.fromarray(image).save(output / "image" / "00000.png")
    Image.fromarray(mask).save(output / "mask" / "00000.png")
    metadata = {
        "source_image": str(Path(args.image).resolve()),
        "source_mask": str(Path(args.mask).resolve()),
        "crop_top": args.crop_top,
        "real_image_count": 1,
        "real_mask_count": 1,
        "condition_encoding": encoding,
        "num_condition_channels": classes,
        "online_sampling": True,
        "fixed_mask_bank": False,
        "important": (
            "Binary input provides only background/annotated-region control. "
            "Independent internal anatomy requires indexed auxiliary labels "
            "0=background, 1=outer target, 2..K=internal structures."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Prepared one real pair at {output}; encoding={encoding}, "
        f"condition channels={classes}. No offline mask bank was created."
    )


if __name__ == "__main__":
    main()
