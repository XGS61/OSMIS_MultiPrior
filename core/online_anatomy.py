"""Online anatomy-condition sampling for one-shot training.

Unlike the previous mask bank, this module creates a fresh condition for every
batch.  It never creates image targets and never treats a warped image as real.
All semantic channels undergo the same smooth global transform.  When auxiliary
internal labels are available (classes >= 2), a small relative transform is
applied inside the globally transformed foreground support.
"""

import math

import torch
import torch.nn.functional as F


class OnlineAnatomySampler:
    def __init__(
        self,
        max_displacement_frac=0.025,
        max_rotation_deg=4.0,
        max_translation_frac=0.03,
        scale_x=(0.90, 1.10),
        scale_y=(0.94, 1.06),
        max_attempts=12,
    ):
        self.max_displacement_frac = max_displacement_frac
        self.max_rotation_rad = math.radians(max_rotation_deg)
        self.max_translation_frac = max_translation_frac
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.max_attempts = max_attempts

    @staticmethod
    def _uniform(count, low, high, device, dtype):
        return low + torch.rand(count, device=device, dtype=dtype) * (high - low)

    def _global_grid(self, count, height, width, device, dtype):
        angle = self._uniform(
            count, -self.max_rotation_rad, self.max_rotation_rad, device, dtype
        )
        sx = self._uniform(count, self.scale_x[0], self.scale_x[1], device, dtype)
        sy = self._uniform(count, self.scale_y[0], self.scale_y[1], device, dtype)
        tx = self._uniform(
            count,
            -2.0 * self.max_translation_frac,
            2.0 * self.max_translation_frac,
            device,
            dtype,
        )
        ty = self._uniform(
            count,
            -2.0 * self.max_translation_frac,
            2.0 * self.max_translation_frac,
            device,
            dtype,
        )
        cosine, sine = torch.cos(angle), torch.sin(angle)
        theta = torch.zeros(count, 2, 3, device=device, dtype=dtype)
        # affine_grid maps output coordinates to input coordinates.
        theta[:, 0, 0] = cosine / sx
        theta[:, 0, 1] = -sine / sx
        theta[:, 1, 0] = sine / sy
        theta[:, 1, 1] = cosine / sy
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        grid = F.affine_grid(
            theta, (count, 1, height, width), align_corners=False
        )

        coarse = torch.randn(count, 2, 5, 5, device=device, dtype=dtype)
        coarse[:, :, 0, :] = 0
        coarse[:, :, -1, :] = 0
        coarse[:, :, :, 0] = 0
        coarse[:, :, :, -1] = 0
        displacement = F.interpolate(
            coarse, size=(height, width), mode="bicubic", align_corners=False
        )
        displacement = torch.tanh(displacement)
        pixel_scale_x = 2.0 * self.max_displacement_frac
        pixel_scale_y = pixel_scale_x * (width / max(height, 1)) * 0.65
        grid = grid + torch.stack(
            (
                displacement[:, 0] * pixel_scale_x,
                displacement[:, 1] * pixel_scale_y,
            ),
            dim=-1,
        )
        return grid

    @staticmethod
    def _hard_one_hot(probabilities):
        labels = probabilities.argmax(dim=1)
        return F.one_hot(
            labels, num_classes=probabilities.shape[1]
        ).permute(0, 3, 1, 2).to(probabilities.dtype)

    def _jitter_internal_labels(self, masks):
        """Move auxiliary internal structures slightly but keep them inside."""
        if masks.shape[1] <= 2:
            return masks
        count, channels, height, width = masks.shape
        foreground_support = 1.0 - masks[:, :1]
        internals = []
        occupied = torch.zeros_like(foreground_support)
        for channel in range(2, channels):
            angle = self._uniform(
                count, -0.025, 0.025, masks.device, masks.dtype
            )
            scale = self._uniform(count, 0.95, 1.05, masks.device, masks.dtype)
            tx = self._uniform(count, -0.025, 0.025, masks.device, masks.dtype)
            ty = self._uniform(count, -0.020, 0.020, masks.device, masks.dtype)
            theta = torch.zeros(
                count, 2, 3, device=masks.device, dtype=masks.dtype
            )
            theta[:, 0, 0] = torch.cos(angle) / scale
            theta[:, 0, 1] = -torch.sin(angle) / scale
            theta[:, 1, 0] = torch.sin(angle) / scale
            theta[:, 1, 1] = torch.cos(angle) / scale
            theta[:, 0, 2] = tx
            theta[:, 1, 2] = ty
            grid = F.affine_grid(
                theta, (count, 1, height, width), align_corners=False
            )
            moved = F.grid_sample(
                masks[:, channel:channel + 1],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            moved = (moved >= 0.5).to(masks.dtype)
            moved = moved * foreground_support * (1.0 - occupied)
            internals.append(moved)
            occupied = torch.clamp(occupied + moved, 0, 1)
        outer = torch.clamp(foreground_support - occupied, 0, 1)
        background = 1.0 - foreground_support
        return torch.cat((background, outer, *internals), dim=1)

    @staticmethod
    def _valid(candidate, reference):
        foreground = 1.0 - candidate[:, :1]
        ref_foreground = 1.0 - reference[:, :1]
        area_ratio = foreground.sum((1, 2, 3)) / ref_foreground.sum(
            (1, 2, 3)
        ).clamp_min(1.0)
        height, width = foreground.shape[2:]
        yy = torch.linspace(
            0, 1, height, device=foreground.device, dtype=foreground.dtype
        )[None, None, :, None]
        xx = torch.linspace(
            0, 1, width, device=foreground.device, dtype=foreground.dtype
        )[None, None, None, :]

        def centroid(mask):
            total = mask.sum((1, 2, 3)).clamp_min(1.0)
            return (
                (mask * xx).sum((1, 2, 3)) / total,
                (mask * yy).sum((1, 2, 3)) / total,
            )

        cx, cy = centroid(foreground)
        ref_cx, ref_cy = centroid(ref_foreground)
        shift = torch.sqrt((cx - ref_cx) ** 2 + (cy - ref_cy) ** 2)
        return (area_ratio >= 0.78) & (area_ratio <= 1.24) & (shift <= 0.07)

    @torch.no_grad()
    def sample(self, base_masks, count=None):
        """Sample fresh hard one-hot conditions on the current device."""
        count = count or base_masks.shape[0]
        reference = base_masks[:1].repeat(count, 1, 1, 1)
        height, width = reference.shape[2:]
        accepted = []
        remaining = count
        attempts = 0
        while remaining > 0 and attempts < self.max_attempts:
            attempts += 1
            source = base_masks[:1].repeat(remaining, 1, 1, 1)
            grid = self._global_grid(
                remaining,
                height,
                width,
                source.device,
                source.dtype,
            )
            warped = F.grid_sample(
                source,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            candidate = self._hard_one_hot(warped)
            candidate = self._jitter_internal_labels(candidate)
            valid = self._valid(candidate, reference[:remaining])
            if valid.any():
                accepted.append(candidate[valid])
                remaining -= int(valid.sum().item())
        if remaining:
            # Conservative fallback remains a valid condition and avoids hangs.
            accepted.append(reference[:remaining])
        return torch.cat(accepted, dim=0)[:count]
