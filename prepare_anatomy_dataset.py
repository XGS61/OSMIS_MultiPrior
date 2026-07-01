"""Build conservative C-plane levator-hiatus pseudo-pairs from one image/mask.

The deformation is deliberately bounded.  It is not intended to simulate a new
clinical state; it only supplies small, smooth shape variations for one-shot
conditional training.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def infer_landmarks(mask):
    """Infer the superior SP and inferior PVM limits from a binary outline."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("The input mask is empty.")
    y_min, y_max = int(ys.min()), int(ys.max())
    band = max(2, int(round((y_max - y_min + 1) * 0.025)))
    sp_x = float(np.median(xs[ys <= y_min + band]))
    pvm_x = float(np.median(xs[ys >= y_max - band]))
    return {"sp": [sp_x, float(y_min)], "pvm": [pvm_x, float(y_max)]}


def row_width(mask, fraction):
    ys, _ = np.nonzero(mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    y = int(round(y_min + fraction * (y_max - y_min)))
    band = mask[max(0, y - 2): min(mask.shape[0], y + 3)]
    _, xs = np.nonzero(band)
    return float(xs.max() - xs.min() + 1) if len(xs) else 0.0


def mask_metrics(mask):
    labels, count = ndimage.label(mask)
    if count:
        sizes = ndimage.sum(mask, labels, range(1, count + 1))
        largest = int(np.argmax(sizes) + 1)
        mask = labels == largest
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot measure an empty mask.")
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    upper_width = row_width(mask, 0.25)
    lower_width = row_width(mask, 0.75)
    landmarks = infer_landmarks(mask)
    return {
        "area": int(mask.sum()),
        "area_ratio_image": float(mask.mean()),
        "bbox_width": int(x_max - x_min + 1),
        "bbox_height": int(y_max - y_min + 1),
        "centroid_x": float(xs.mean()),
        "centroid_y": float(ys.mean()),
        "upper_width": upper_width,
        "lower_width": lower_width,
        "pear_ratio": float(upper_width / max(lower_width, 1.0)),
        "components": int(count),
        "sp": landmarks["sp"],
        "pvm": landmarks["pvm"],
    }


def make_smooth_field(shape, rng, max_dx, max_dy, grid_size):
    """Create a border-anchored cubic field similar to a coarse B-spline warp."""
    h, w = shape
    ctrl_y = rng.normal(0.0, max_dy, size=(grid_size, grid_size))
    ctrl_x = rng.normal(0.0, max_dx, size=(grid_size, grid_size))
    ctrl_y[[0, -1], :] = 0
    ctrl_y[:, [0, -1]] = 0
    ctrl_x[[0, -1], :] = 0
    ctrl_x[:, [0, -1]] = 0
    zoom = (h / grid_size, w / grid_size)
    field_y = ndimage.zoom(ctrl_y, zoom, order=3)[:h, :w]
    field_x = ndimage.zoom(ctrl_x, zoom, order=3)[:h, :w]
    sigma = max(h, w) / 45.0
    field_y = ndimage.gaussian_filter(field_y, sigma=sigma, mode="nearest")
    field_x = ndimage.gaussian_filter(field_x, sigma=sigma, mode="nearest")
    return field_y, field_x


def warp(array, field_y, field_x, order):
    h, w = array.shape[:2]
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = [yy + field_y, xx + field_x]
    if array.ndim == 2:
        return ndimage.map_coordinates(array, coords, order=order, mode="reflect")
    channels = [
        ndimage.map_coordinates(array[..., c], coords, order=order, mode="reflect")
        for c in range(array.shape[2])
    ]
    return np.stack(channels, axis=-1)


def ratio(value, reference):
    return float(value) / max(float(reference), 1e-6)


def validate(candidate, reference_metrics, constraints):
    labels, count = ndimage.label(candidate)
    if count != 1:
        return False, "not_single_component", None
    filled = ndimage.binary_fill_holes(candidate)
    hole_fraction = float((filled & ~candidate).sum()) / max(float(filled.sum()), 1.0)
    if hole_fraction > constraints["max_hole_fraction"]:
        return False, "holes", None
    metrics = mask_metrics(candidate)
    checks = {
        "area": constraints["area_ratio"][0]
        <= ratio(metrics["area"], reference_metrics["area"])
        <= constraints["area_ratio"][1],
        "width": constraints["width_ratio"][0]
        <= ratio(metrics["bbox_width"], reference_metrics["bbox_width"])
        <= constraints["width_ratio"][1],
        "height": constraints["height_ratio"][0]
        <= ratio(metrics["bbox_height"], reference_metrics["bbox_height"])
        <= constraints["height_ratio"][1],
        "centroid": np.hypot(
            metrics["centroid_x"] - reference_metrics["centroid_x"],
            metrics["centroid_y"] - reference_metrics["centroid_y"],
        )
        <= constraints["max_centroid_shift_px"],
        "pear": constraints["pear_ratio"][0]
        <= metrics["pear_ratio"]
        <= constraints["pear_ratio"][1],
        "sp": np.linalg.norm(np.asarray(metrics["sp"]) - np.asarray(reference_metrics["sp"]))
        <= constraints["max_landmark_shift_px"],
        "pvm": np.linalg.norm(np.asarray(metrics["pvm"]) - np.asarray(reference_metrics["pvm"]))
        <= constraints["max_landmark_shift_px"],
    }
    failed = [name for name, ok in checks.items() if not ok]
    return not failed, ",".join(failed), metrics


def save_pair(image, mask, root, index):
    name = f"{index:05d}.png"
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB").save(
        root / "image" / name
    )
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(root / "mask" / name)


def parse_point(value):
    if value is None:
        return None
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Points must use x,y format.")
    return [float(parts[0]), float(parts[1])]


def main():
    parser = argparse.ArgumentParser(
        description="Generate anatomy-constrained C-plane pseudo image/mask pairs."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-variants", type=int, default=32)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--max-displacement-frac", type=float, default=0.035)
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument("--sp", type=parse_point, help="optional SP point in x,y pixels")
    parser.add_argument("--pvm", type=parse_point, help="optional PVM point in x,y pixels")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32)
    mask = np.asarray(Image.open(args.mask).convert("L")) >= 128
    if image.shape[:2] != mask.shape:
        raise ValueError(f"Image/mask mismatch: {image.shape[:2]} vs {mask.shape}")

    output = Path(args.output)
    if output.exists() and any(output.rglob("*.png")) and not args.overwrite:
        raise FileExistsError(
            f"{output} already contains PNG files. Pass --overwrite to replace them."
        )
    for subdir in ("image", "mask"):
        path = output / subdir
        path.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            for old in path.glob("*.png"):
                old.unlink()

    ref = mask_metrics(mask)
    if args.sp is not None:
        ref["sp"] = args.sp
    if args.pvm is not None:
        ref["pvm"] = args.pvm
    min_dim = min(mask.shape)
    constraints = {
        "area_ratio": [0.86, 1.14],
        "width_ratio": [0.88, 1.12],
        "height_ratio": [0.93, 1.07],
        "pear_ratio": [
            max(1.35, ref["pear_ratio"] * 0.75),
            ref["pear_ratio"] * 1.30,
        ],
        "max_centroid_shift_px": float(min_dim * 0.035),
        "max_landmark_shift_px": float(min_dim * 0.045),
        "max_hole_fraction": 0.002,
    }

    rng = np.random.default_rng(args.seed)
    records = []
    save_pair(image, mask, output, 0)
    records.append({"index": 0, "kind": "reference", "metrics": ref})

    attempts = 0
    while len(records) < args.num_variants and attempts < args.max_attempts:
        attempts += 1
        amp = min_dim * args.max_displacement_frac
        field_y, field_x = make_smooth_field(
            mask.shape,
            rng,
            max_dx=amp,
            max_dy=amp * 0.65,
            grid_size=args.grid_size,
        )
        candidate = warp(mask.astype(np.float32), field_y, field_x, order=0) >= 0.5
        candidate = ndimage.binary_fill_holes(candidate)
        valid, reason, metrics = validate(candidate, ref, constraints)
        if not valid:
            continue
        warped_image = warp(image, field_y, field_x, order=1)
        index = len(records)
        save_pair(warped_image, candidate, output, index)
        records.append(
            {
                "index": index,
                "kind": "smooth_bspline_like_warp",
                "metrics": metrics,
                "field_max_abs_x": float(np.abs(field_x).max()),
                "field_max_abs_y": float(np.abs(field_y).max()),
            }
        )

    if len(records) < args.num_variants:
        raise RuntimeError(
            f"Only generated {len(records)}/{args.num_variants} valid samples "
            f"after {attempts} attempts. Relax constraints deliberately if needed."
        )

    metadata = {
        "source_image": str(Path(args.image).resolve()),
        "source_mask": str(Path(args.mask).resolve()),
        "segmentation_target": "interior region of the levator hiatus on the C-plane",
        "landmark_interpretation": {
            "sp": "superior limit: posterior aspect of the symphysis pubis",
            "pvm": "inferior limit: anterior border of the pubovisceral muscle",
        },
        "clinical_scope": (
            "Conservative within-state variation only; not a simulator for transitions "
            "between rest, contraction, and Valsalva."
        ),
        "constraints": constraints,
        "reference_metrics": ref,
        "accepted": len(records),
        "attempts": attempts,
        "samples": records,
    }
    with open(output / "anatomy_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(f"Prepared {len(records)} pairs at {output}")
    print(f"Accepted {len(records) - 1} deformations from {attempts} attempts")
    print(f"Inferred SP={ref['sp']}, PVM={ref['pvm']}")


if __name__ == "__main__":
    main()
