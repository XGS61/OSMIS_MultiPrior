"""Prepare one real C-plane ultrasound pair and a mask-only prior bank.

The real image domain intentionally contains exactly one image and one mask.
Smooth deformations are written only to ``mask_priors`` and are never used as
real image reconstruction targets.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def largest_component(mask):
    labels, count = ndimage.label(mask)
    if count == 0:
        raise ValueError("The input mask is empty.")
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return labels == int(np.argmax(sizes) + 1)


def row_width(mask, fraction):
    ys, _ = np.nonzero(mask)
    y = int(round(ys.min() + fraction * (ys.max() - ys.min())))
    _, xs = np.nonzero(mask[max(0, y - 2): min(mask.shape[0], y + 3)])
    return float(xs.max() - xs.min() + 1) if len(xs) else 0.0


def metrics(mask):
    mask = largest_component(mask)
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    upper = row_width(mask, 0.25)
    lower = row_width(mask, 0.75)
    return {
        "area": int(mask.sum()),
        "bbox_width": int(x1 - x0 + 1),
        "bbox_height": int(y1 - y0 + 1),
        "centroid_x": float(xs.mean()),
        "centroid_y": float(ys.mean()),
        "pear_ratio": float(upper / max(lower, 1.0)),
        "upper_width": upper,
        "lower_width": lower,
    }


def smooth_field(shape, rng, amplitude, grid_size):
    h, w = shape
    control = rng.normal(0.0, 1.0, size=(2, grid_size, grid_size))
    control[:, [0, -1], :] = 0
    control[:, :, [0, -1]] = 0
    fields = []
    for axis in range(2):
        field = ndimage.zoom(
            control[axis], (h / grid_size, w / grid_size), order=3
        )[:h, :w]
        field = ndimage.gaussian_filter(field, sigma=max(h, w) / 50.0)
        peak = max(float(np.abs(field).max()), 1e-6)
        fields.append(field / peak * amplitude)
    fields[0] *= 0.65
    return fields


def deform_mask(mask, rng, max_displacement, grid_size):
    h, w = mask.shape
    dy, dx = smooth_field(mask.shape, rng, max_displacement, grid_size)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    warped = ndimage.map_coordinates(
        mask.astype(np.float32), [yy + dy, xx + dx], order=1, mode="constant"
    )
    warped = warped >= 0.5

    # A small global affine component provides controlled size/position change.
    zoom_y = rng.uniform(0.94, 1.06)
    zoom_x = rng.uniform(0.90, 1.10)
    angle = rng.uniform(-4.0, 4.0)
    candidate = ndimage.zoom(warped.astype(np.uint8), (zoom_y, zoom_x), order=0)
    candidate = ndimage.rotate(candidate, angle, reshape=False, order=0)
    canvas = np.zeros_like(mask, dtype=bool)
    ch, cw = candidate.shape
    shift_y = int(rng.uniform(-0.025, 0.025) * h)
    shift_x = int(rng.uniform(-0.035, 0.035) * w)
    dst_y = (h - ch) // 2 + shift_y
    dst_x = (w - cw) // 2 + shift_x
    sy0, sx0 = max(0, -dst_y), max(0, -dst_x)
    dy0, dx0 = max(0, dst_y), max(0, dst_x)
    copy_h = min(ch - sy0, h - dy0)
    copy_w = min(cw - sx0, w - dx0)
    if copy_h > 0 and copy_w > 0:
        canvas[dy0:dy0 + copy_h, dx0:dx0 + copy_w] = (
            candidate[sy0:sy0 + copy_h, sx0:sx0 + copy_w] > 0
        )
    return ndimage.binary_fill_holes(canvas)


def valid(candidate, reference, min_dim):
    labels, count = ndimage.label(candidate)
    if count != 1:
        return False, None
    filled = ndimage.binary_fill_holes(candidate)
    if np.any(filled != candidate):
        return False, None
    cur = metrics(candidate)
    area = cur["area"] / reference["area"]
    width = cur["bbox_width"] / reference["bbox_width"]
    height = cur["bbox_height"] / reference["bbox_height"]
    shift = np.hypot(
        cur["centroid_x"] - reference["centroid_x"],
        cur["centroid_y"] - reference["centroid_y"],
    )
    pear_low = max(1.15, reference["pear_ratio"] * 0.72)
    pear_high = reference["pear_ratio"] * 1.38
    ok = (
        0.78 <= area <= 1.24
        and 0.84 <= width <= 1.18
        and 0.88 <= height <= 1.12
        and shift <= 0.055 * min_dim
        and pear_low <= cur["pear_ratio"] <= pear_high
    )
    return ok, cur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-mask-priors", type=int, default=64)
    parser.add_argument("--crop-top", type=int, default=20)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--max-attempts", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"))
    mask = np.asarray(Image.open(args.mask).convert("L")) >= 128
    if image.shape[:2] != mask.shape:
        raise ValueError(f"Image/mask size mismatch: {image.shape[:2]} vs {mask.shape}")
    if args.crop_top:
        image = image[args.crop_top:]
        mask = mask[args.crop_top:]
    mask = largest_component(mask)
    mask = ndimage.binary_fill_holes(mask)

    root = Path(args.output)
    if root.exists() and args.overwrite:
        shutil.rmtree(root)
    for name in ("image", "mask", "mask_priors"):
        (root / name).mkdir(parents=True, exist_ok=True)
    if any((root / "image").glob("*")) and not args.overwrite:
        raise FileExistsError(f"{root} is not empty; pass --overwrite.")

    Image.fromarray(image).save(root / "image" / "00000.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(root / "mask" / "00000.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(root / "mask_priors" / "00000.png")

    reference = metrics(mask)
    rng = np.random.default_rng(args.seed)
    records = [{"index": 0, "kind": "original_mask", "metrics": reference}]
    attempts = 0
    min_dim = min(mask.shape)
    while len(records) < args.num_mask_priors and attempts < args.max_attempts:
        attempts += 1
        candidate = deform_mask(
            mask, rng, max_displacement=0.035 * min_dim, grid_size=5
        )
        accepted, cur_metrics = valid(candidate, reference, min_dim)
        if not accepted:
            continue
        # Reject near duplicates in area/centroid space.
        signature = np.array([
            cur_metrics["area"] / reference["area"],
            (cur_metrics["centroid_x"] - reference["centroid_x"]) / min_dim,
            (cur_metrics["centroid_y"] - reference["centroid_y"]) / min_dim,
            cur_metrics["pear_ratio"] / reference["pear_ratio"],
        ])
        if any(
            np.linalg.norm(signature - np.asarray(item["signature"])) < 0.006
            for item in records[1:]
        ):
            continue
        index = len(records)
        Image.fromarray(candidate.astype(np.uint8) * 255).save(
            root / "mask_priors" / f"{index:05d}.png"
        )
        records.append({
            "index": index,
            "kind": "anatomy_bounded_mask_prior",
            "metrics": cur_metrics,
            "signature": signature.tolist(),
        })

    if len(records) != args.num_mask_priors:
        raise RuntimeError(
            f"Generated only {len(records)}/{args.num_mask_priors} masks "
            f"after {attempts} attempts."
        )

    metadata = {
        "source_image": str(Path(args.image).resolve()),
        "source_mask": str(Path(args.mask).resolve()),
        "crop_top": args.crop_top,
        "real_image_count": 1,
        "real_mask_count": 1,
        "mask_prior_count": len(records),
        "critical_note": (
            "Only image/00000.png is a real training image. mask_priors contains "
            "conditions only; no warped image is treated as a real target."
        ),
        "segmentation_target": "levator hiatus interior on the C-plane",
        "reference_metrics": reference,
        "samples": records,
    }
    with open(root / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"Prepared exactly 1 real pair and {len(records)} mask-only priors at {root}")


if __name__ == "__main__":
    main()
