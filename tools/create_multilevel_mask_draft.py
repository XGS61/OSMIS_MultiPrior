"""Create a review-only hierarchical annotation draft for one pelvic case.

This utility preserves the supplied levator-hiatus annotation and derives:
1) the rendered ultrasound support;
2) three position-based dark-structure candidates inside the supplied mask;
3) boundary and distance-map conditions.

The internal candidates are image-processing proposals, not clinical labels.
They must be reviewed before being used as semantic classes.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


PALETTE = np.asarray(
    [
        [0, 0, 0],        # 0: outside rendered support
        [80, 80, 80],     # 1: rendered tissue/support
        [0, 210, 255],    # 2: supplied levator-hiatus region
        [255, 60, 60],    # 3: anterior internal candidate
        [80, 255, 100],   # 4: middle internal candidate
        [200, 80, 255],   # 5: posterior internal candidate
    ],
    dtype=np.uint8,
)


def largest_component(mask):
    labels, count = ndi.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    areas = ndi.sum(mask, labels, range(1, count + 1))
    return labels == (int(np.argmax(areas)) + 1)


def rendered_support(rgb):
    value = rgb.max(axis=2)
    support = value > 12
    support = ndi.binary_closing(support, iterations=4)
    support = ndi.binary_opening(support, iterations=2)
    support = largest_component(support)
    return ndi.binary_fill_holes(support)


def dark_candidate(gray, target, box, percentile, expected_center):
    height, width = gray.shape
    x0, y0, x1, y1 = box
    x0, x1 = int(x0 * width), int(x1 * width)
    y0, y1 = int(y0 * height), int(y1 * height)
    roi = np.zeros_like(target, dtype=bool)
    roi[y0:y1, x0:x1] = True
    valid = roi & target
    if not valid.any():
        return np.zeros_like(target, dtype=bool)

    threshold = np.percentile(gray[valid], percentile)
    proposal = (gray <= threshold) & valid
    proposal = ndi.binary_opening(proposal, iterations=1)
    proposal = ndi.binary_closing(proposal, iterations=2)

    labels, count = ndi.label(proposal)
    if count == 0:
        return proposal
    expected_x = expected_center[0] * width
    expected_y = expected_center[1] * height
    best_label, best_score = 0, float("inf")
    for label in range(1, count + 1):
        ys, xs = np.where(labels == label)
        if xs.size < 20:
            continue
        distance = (xs.mean() - expected_x) ** 2 + (ys.mean() - expected_y) ** 2
        score = distance / max(xs.size, 1) ** 0.5
        if score < best_score:
            best_label, best_score = label, score
    if best_label == 0:
        return np.zeros_like(target, dtype=bool)
    candidate = labels == best_label
    candidate = ndi.binary_closing(candidate, iterations=3)
    candidate = ndi.binary_fill_holes(candidate)
    candidate = ndi.binary_dilation(candidate, iterations=1)
    return candidate & target


def fallback_candidate(target, expected_center, radius=(0.055, 0.035), occupied=None):
    """Small geometry prior used only when dark-component detection fails."""
    if occupied is None:
        occupied = np.zeros_like(target, dtype=bool)
    valid = target & ~occupied
    if not valid.any():
        return np.zeros_like(target, dtype=bool)
    height, width = target.shape
    ys, xs = np.where(valid)
    expected_x = expected_center[0] * width
    expected_y = expected_center[1] * height
    nearest = np.argmin((xs - expected_x) ** 2 + (ys - expected_y) ** 2)
    cx, cy = xs[nearest], ys[nearest]
    yy, xx = np.ogrid[:height, :width]
    rx = max(4.0, radius[0] * width)
    ry = max(4.0, radius[1] * height)
    candidate = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    candidate = ndi.binary_closing(candidate & valid, iterations=2)
    candidate = ndi.binary_fill_holes(candidate)
    return candidate & valid


def ensure_candidate(candidate, target, expected_center, occupied=None):
    if candidate.sum() >= 20:
        return candidate
    return fallback_candidate(target, expected_center, occupied=occupied)


def save_binary(path, mask):
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    target = np.asarray(Image.open(args.mask).convert("L")) >= 128
    if rgb.shape[:2] != target.shape:
        raise ValueError(f"Image/mask mismatch: {rgb.shape[:2]} versus {target.shape}")

    gray = rgb.astype(np.float32).mean(axis=2)
    support = rendered_support(rgb)
    target &= support

    # Position-based proposals only. Names intentionally avoid anatomical claims.
    anterior = dark_candidate(
        gray, target, (0.38, 0.23, 0.58, 0.39), 28, (0.49, 0.31)
    )
    anterior = ensure_candidate(anterior, target, (0.49, 0.31))
    middle = dark_candidate(
        gray, target, (0.34, 0.50, 0.58, 0.68), 24, (0.45, 0.59)
    )
    middle = ensure_candidate(middle, target, (0.45, 0.59), anterior)
    posterior = dark_candidate(
        gray, target, (0.42, 0.64, 0.60, 0.78), 18, (0.53, 0.70)
    )
    middle &= ~anterior
    middle = ensure_candidate(middle, target, (0.45, 0.59), anterior)
    posterior &= ~(anterior | middle)
    posterior = ensure_candidate(
        posterior, target, (0.53, 0.70), anterior | middle
    )

    indexed = np.zeros(target.shape, dtype=np.uint8)
    indexed[support] = 1
    indexed[target] = 2
    indexed[anterior] = 3
    indexed[middle] = 4
    indexed[posterior] = 5

    boundary = target ^ ndi.binary_erosion(target, iterations=4)
    inside_distance = ndi.distance_transform_edt(target)
    outside_distance = ndi.distance_transform_edt(~target)
    signed_distance = inside_distance - outside_distance
    distance_scale = max(float(np.abs(signed_distance).max()), 1.0)
    distance_u8 = np.clip(
        (signed_distance / distance_scale + 1.0) * 127.5, 0, 255
    ).astype(np.uint8)

    color = PALETTE[indexed]
    overlay = (rgb.astype(np.float32) * 0.58 + color.astype(np.float32) * 0.42)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    Image.fromarray(indexed).save(output / "multilevel_mask_indexed.png")
    Image.fromarray(color).save(output / "multilevel_mask_color.png")
    Image.fromarray(overlay).save(output / "multilevel_mask_overlay.png")
    save_binary(output / "channel_0_rendered_support.png", support)
    save_binary(output / "channel_1_levator_hiatus.png", target)
    save_binary(output / "channel_2_internal_anterior_candidate.png", anterior)
    save_binary(output / "channel_3_internal_middle_candidate.png", middle)
    save_binary(output / "channel_4_internal_posterior_candidate.png", posterior)
    save_binary(output / "channel_5_hiatus_boundary.png", boundary)
    Image.fromarray(distance_u8).save(output / "channel_6_hiatus_signed_distance.png")

    legend_height = 118
    panel = Image.new("RGB", (rgb.shape[1] * 3, rgb.shape[0] + legend_height), "white")
    panel.paste(Image.fromarray(rgb), (0, 0))
    panel.paste(Image.fromarray(overlay), (rgb.shape[1], 0))
    panel.paste(Image.fromarray(color), (rgb.shape[1] * 2, 0))
    draw = ImageDraw.Draw(panel)
    labels = [
        "0 outside support",
        "1 rendered tissue",
        "2 supplied hiatus",
        "3 anterior candidate",
        "4 middle candidate",
        "5 posterior candidate",
    ]
    for index, label in enumerate(labels):
        row, column = divmod(index, 3)
        x = 12 + column * rgb.shape[1]
        y = rgb.shape[0] + 12 + row * 42
        draw.rectangle((x, y, x + 24, y + 24), fill=tuple(PALETTE[index]))
        draw.text((x + 34, y + 4), label, fill=(0, 0, 0))
    panel.save(output / "review_panel.png")

    np.savez_compressed(
        output / "hierarchical_conditions.npz",
        rendered_support=support.astype(np.uint8),
        levator_hiatus=target.astype(np.uint8),
        internal_anterior_candidate=anterior.astype(np.uint8),
        internal_middle_candidate=middle.astype(np.uint8),
        internal_posterior_candidate=posterior.astype(np.uint8),
        hiatus_boundary=boundary.astype(np.uint8),
        hiatus_signed_distance=signed_distance.astype(np.float32),
    )
    print(f"Saved review-only annotation draft to {output}")


if __name__ == "__main__":
    main()
