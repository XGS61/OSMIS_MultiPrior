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
        max_displacement_frac=0.04,
        max_rotation_deg=7.0,
        max_translation_frac=0.045,
        scale_x=(0.84, 1.16),
        scale_y=(0.88, 1.12),
        support_max_displacement_frac=0.010,
        support_max_rotation_deg=1.5,
        support_max_translation_frac=0.010,
        support_scale_x=(0.97, 1.03),
        support_scale_y=(0.98, 1.02),
        max_attempts=18,
    ):
        self.max_displacement_frac = max_displacement_frac
        self.max_rotation_rad = math.radians(max_rotation_deg)
        self.max_translation_frac = max_translation_frac
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.support_max_displacement_frac = support_max_displacement_frac
        self.support_max_rotation_rad = math.radians(support_max_rotation_deg)
        self.support_max_translation_frac = support_max_translation_frac
        self.support_scale_x = support_scale_x
        self.support_scale_y = support_scale_y
        self.max_attempts = max_attempts

    @staticmethod
    def _uniform(count, low, high, device, dtype):
        return low + torch.rand(count, device=device, dtype=dtype) * (high - low)

    def _make_grid(
        self,
        count,
        height,
        width,
        device,
        dtype,
        max_rotation_rad,
        max_translation_frac,
        scale_x,
        scale_y,
        max_displacement_frac,
    ):
        angle = self._uniform(
            count, -max_rotation_rad, max_rotation_rad, device, dtype
        )
        sx = self._uniform(count, scale_x[0], scale_x[1], device, dtype)
        sy = self._uniform(count, scale_y[0], scale_y[1], device, dtype)
        tx = self._uniform(
            count,
            -2.0 * max_translation_frac,
            2.0 * max_translation_frac,
            device,
            dtype,
        )
        ty = self._uniform(
            count,
            -2.0 * max_translation_frac,
            2.0 * max_translation_frac,
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
        pixel_scale_x = 2.0 * max_displacement_frac
        pixel_scale_y = pixel_scale_x * (width / max(height, 1)) * 0.65
        grid = grid + torch.stack(
            (
                displacement[:, 0] * pixel_scale_x,
                displacement[:, 1] * pixel_scale_y,
            ),
            dim=-1,
        )
        return grid

    def _global_grid(self, count, height, width, device, dtype):
        return self._make_grid(
            count,
            height,
            width,
            device,
            dtype,
            self.max_rotation_rad,
            self.max_translation_frac,
            self.scale_x,
            self.scale_y,
            self.max_displacement_frac,
        )

    def _support_grid(self, count, height, width, device, dtype):
        return self._make_grid(
            count,
            height,
            width,
            device,
            dtype,
            self.support_max_rotation_rad,
            self.support_max_translation_frac,
            self.support_scale_x,
            self.support_scale_y,
            self.support_max_displacement_frac,
        )

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

    def _jitter_hierarchical_internal_labels(self, internals, target):
        """Move internal candidate labels while keeping them inside target."""
        count, channels, height, width = internals.shape
        moved_channels = []
        occupied = torch.zeros_like(target)
        for channel in range(channels):
            angle = self._uniform(
                count, -0.052, 0.052, internals.device, internals.dtype
            )
            scale = self._uniform(count, 0.88, 1.12, internals.device, internals.dtype)
            tx = self._uniform(count, -0.035, 0.035, internals.device, internals.dtype)
            ty = self._uniform(count, -0.030, 0.030, internals.device, internals.dtype)
            theta = torch.zeros(
                count, 2, 3, device=internals.device, dtype=internals.dtype
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
                internals[:, channel:channel + 1],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            moved = (moved >= 0.5).to(internals.dtype)
            moved = moved * target * (1.0 - occupied)
            moved_channels.append(moved)
            occupied = torch.clamp(occupied + moved, 0, 1)
        return torch.cat(moved_channels, dim=1)

    def _rebuild_hierarchical(self, candidate, support_override=None):
        """Rebuild 0/1/2/3/4/5 regions after a global deformation.

        Input convention:
          0 background
          1 rendered support outside target
          2 target remainder
          3..5 internal candidates
        Output preserves the same exclusive layout.  Internal candidate classes
        get a small relative jitter, but remain inside the target union.
        """
        support = (
            support_override
            if support_override is not None
            else candidate[:, 1:].sum(dim=1, keepdim=True).clamp(0, 1)
        )
        support = support.clamp(0, 1)
        target = candidate[:, 2:].sum(dim=1, keepdim=True).clamp(0, 1) * support
        internals = self._jitter_hierarchical_internal_labels(
            candidate[:, 3:], target
        )
        occupied = internals.sum(dim=1, keepdim=True).clamp(0, 1)
        background = 1.0 - support
        support_not_target = (support - target).clamp(0, 1)
        target_remainder = (target - occupied).clamp(0, 1)
        return torch.cat(
            (background, support_not_target, target_remainder, internals),
            dim=1,
        )

    @staticmethod
    def _valid(candidate, reference):
        if candidate.shape[1] > 2:
            foreground = candidate[:, 2:].sum(dim=1, keepdim=True).clamp(0, 1)
            ref_foreground = reference[:, 2:].sum(dim=1, keepdim=True).clamp(0, 1)
        else:
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
        # The C-plane levator-hiatus outline can vary substantially in area,
        # width, and pear-like taper across acquisition/subject state.  Keep
        # the sampler visibly non-conservative while rejecting implausible
        # jumps that would move the annotated anatomy out of the rendered
        # support or destroy the target topology.
        valid = (area_ratio >= 0.68) & (area_ratio <= 1.42) & (shift <= 0.10)
        if candidate.shape[1] > 2:
            support = candidate[:, 1:].sum(dim=1, keepdim=True).clamp(0, 1)
            ref_support = reference[:, 1:].sum(dim=1, keepdim=True).clamp(0, 1)
            support_area_ratio = support.sum((1, 2, 3)) / ref_support.sum(
                (1, 2, 3)
            ).clamp_min(1.0)
            support_cx, support_cy = centroid(support)
            ref_support_cx, ref_support_cy = centroid(ref_support)
            support_shift = torch.sqrt(
                (support_cx - ref_support_cx) ** 2
                + (support_cy - ref_support_cy) ** 2
            )
            valid = valid & (support_area_ratio >= 0.90)
            valid = valid & (support_area_ratio <= 1.12)
            valid = valid & (support_shift <= 0.045)
        if candidate.shape[1] > 3:
            internals = candidate[:, 3:]
            ref_internals = reference[:, 3:]
            internal_ratio = internals.sum((2, 3)) / ref_internals.sum(
                (2, 3)
            ).clamp_min(1.0)
            valid = valid & (internal_ratio >= 0.35).all(1)
            valid = valid & (internal_ratio <= 2.10).all(1)
        return valid

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
            anatomy_grid = self._global_grid(
                remaining,
                height,
                width,
                source.device,
                source.dtype,
            )
            warped = F.grid_sample(
                source,
                anatomy_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            candidate = self._hard_one_hot(warped)
            if candidate.shape[1] > 2:
                support_source = source[:, 1:].sum(dim=1, keepdim=True).clamp(0, 1)
                support_grid = self._support_grid(
                    remaining,
                    height,
                    width,
                    source.device,
                    source.dtype,
                )
                weak_support = F.grid_sample(
                    support_source,
                    support_grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                weak_support = (weak_support >= 0.5).to(source.dtype)
                candidate = self._rebuild_hierarchical(
                    candidate, support_override=weak_support
                )
            else:
                candidate = self._jitter_internal_labels(candidate)
            valid = self._valid(candidate, reference[:remaining])
            if valid.any():
                accepted.append(candidate[valid])
                remaining -= int(valid.sum().item())
        if remaining:
            # Conservative fallback remains a valid condition and avoids hangs.
            accepted.append(reference[:remaining])
        return torch.cat(accepted, dim=0)[:count]
