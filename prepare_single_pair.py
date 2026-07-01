import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description="Prepare one RGB image and binary mask for OSMIS."
    )
    parser.add_argument("--image", required=True, help="source image")
    parser.add_argument("--mask", required=True, help="binary segmentation mask")
    parser.add_argument("--dataset-name", default="rendered_us_3d_1")
    parser.add_argument("--dataroot", default="datasets")
    parser.add_argument(
        "--resize-mask",
        action="store_true",
        help="resize a mismatched mask to the image using nearest-neighbor interpolation",
    )
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    mask = Image.open(args.mask).convert("L")
    if image.size != mask.size:
        if not args.resize_mask:
            raise ValueError(
                f"Image/mask size mismatch: {image.size} vs {mask.size}. "
                "Use --resize-mask only when the masks are known to share the same geometry."
            )
        mask = mask.resize(image.size, Image.Resampling.NEAREST)

    mask_array = np.asarray(mask)
    binary_mask = np.where(mask_array >= 128, 255, 0).astype(np.uint8)

    root = Path(args.dataroot) / args.dataset_name
    image_dir = root / "image"
    mask_dir = root / "mask"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / "00000.png"
    mask_path = mask_dir / "00000.png"
    image.save(image_path)
    Image.fromarray(binary_mask, mode="L").save(mask_path)

    print(f"image={image_path}")
    print(f"mask={mask_path}")
    print(f"size={image.width}x{image.height}")
    print(f"foreground_ratio={(binary_mask > 0).mean():.6f}")


if __name__ == "__main__":
    main()
