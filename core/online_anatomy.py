"""Topology-aware online sampling for hierarchical pelvic-floor conditions.

The sampler is deliberately non-neural: one annotated case cannot identify a
population shape distribution. It applies bounded smooth deformations to the
supplied hierarchy, then enforces containment, ordering, area, and centroid
constraints. It creates conditions only and never creates pseudo-real images.
"""

import math

import torch
import torch.nn.functional as F


class OnlineAnatomySampler:
    SUPPORT = 0
    HIATUS = 1
    INTERNAL_START = 2
    INTERNAL_END = 5
    BOUNDARY = 5
    DISTANCE = 6

    def __init__(
        self,
        max_displacement_frac=0.025,
        max_rotation_deg=4.0,
        max_translation_frac=0.03,
        scale_x=(0.92, 1.08),
        scale_y=(0.94, 1.06),
        max_attempts=12,
        distance_steps=12,
    ):
        self.max_displacement_frac = max_displacement_frac
        self.max_rotation_rad = math.radians(max_rotation_deg)
        self.max_translation_frac = max_translation_frac
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.max_attempts = max_attempts
        self.distance_steps = distance_steps

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
        coarse[:, :, (0, -1), :] = 0
        coarse[:, :, :, (0, -1)] = 0
        displacement = torch.tanh(
            F.interpolate(
                coarse,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )
        )
        grid = grid + torch.stack(
            (
                displacement[:, 0] * (2.0 * self.max_displacement_frac),
                displacement[:, 1]
                * (2.0 * self.max_displacement_frac)
                * (width / max(height, 1))
                * 0.65,
            ),
            dim=-1,
        )
        return grid

    def _relative_internal_jitter(self, internals, target):
        count, channels, height, width = internals.shape
        moved_channels = []
        occupied = torch.zeros_like(target)
        for channel in range(channels):
            angle = self._uniform(
                count, -0.025, 0.025, internals.device, internals.dtype
            )
            scale = self._uniform(
                count, 0.94, 1.06, internals.device, internals.dtype
            )
            tx = self._uniform(
                count, -0.020, 0.020, internals.device, internals.dtype
            )
            ty = self._uniform(
                count, -0.018, 0.018, internals.device, internals.dtype
            )
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

    @staticmethod
    def _boundary(mask, width=4):
        eroded = -F.max_pool2d(-mask, width * 2 + 1, stride=1, padding=width)
        return (mask - eroded).clamp(0, 1)

    def _signed_distance(self, mask):
        inside = mask
        outside = 1.0 - mask
        inside_score = torch.zeros_like(mask)
        outside_score = torch.zeros_like(mask)
        for _ in range(self.distance_steps):
            inside_score = inside_score + inside
            outside_score = outside_score + outside
            inside = -F.max_pool2d(-inside, 3, stride=1, padding=1)
            outside = -F.max_pool2d(-outside, 3, stride=1, padding=1)
        return ((inside_score - outside_score) / self.distance_steps).clamp(-1, 1)

    def _rebuild(self, support, target, internals):
        support = (support >= 0.5).to(target.dtype)
        target = (target >= 0.5).to(target.dtype) * support
        internals = self._relative_internal_jitter(
            (internals >= 0.5).to(target.dtype), target
        )
        occupied = internals.sum(dim=1, keepdim=True).clamp(0, 1)
        region_masks = torch.cat(
            (
                1.0 - support,
                support * (1.0 - target),
                target * (1.0 - occupied),
                internals,
            ),
            dim=1,
        )
        conditions = torch.cat(
            (
                support,
                target,
                internals,
                self._boundary(target),
                self._signed_distance(target),
            ),
            dim=1,
        )
        return conditions, region_masks

    @staticmethod
    def _centroid(mask):
        height, width = mask.shape[2:]
        yy = torch.linspace(
            0, 1, height, device=mask.device, dtype=mask.dtype
        )[None, None, :, None]
        xx = torch.linspace(
            0, 1, width, device=mask.device, dtype=mask.dtype
        )[None, None, None, :]
        total = mask.sum((1, 2, 3)).clamp_min(1.0)
        return (
            (mask * xx).sum((1, 2, 3)) / total,
            (mask * yy).sum((1, 2, 3)) / total,
        )

    def _valid(self, conditions, reference):
        target = conditions[:, self.HIATUS:self.HIATUS + 1]
        ref_target = reference[:, self.HIATUS:self.HIATUS + 1]
        area_ratio = target.sum((1, 2, 3)) / ref_target.sum(
            (1, 2, 3)
        ).clamp_min(1.0)
        cx, cy = self._centroid(target)
        ref_cx, ref_cy = self._centroid(ref_target)
        shift = torch.sqrt((cx - ref_cx) ** 2 + (cy - ref_cy) ** 2)
        valid = (area_ratio >= 0.80) & (area_ratio <= 1.22) & (shift <= 0.07)

        internals = conditions[:, self.INTERNAL_START:self.INTERNAL_END]
        ref_internals = reference[:, self.INTERNAL_START:self.INTERNAL_END]
        internal_ratio = internals.sum((2, 3)) / ref_internals.sum(
            (2, 3)
        ).clamp_min(1.0)
        valid = valid & (internal_ratio >= 0.65).all(1)
        valid = valid & (internal_ratio <= 1.45).all(1)
        centroids_y = torch.stack(
            [self._centroid(internals[:, i:i + 1])[1] for i in range(3)],
            dim=1,
        )
        valid = valid & (centroids_y[:, 0] < centroids_y[:, 1])
        valid = valid & (centroids_y[:, 1] < centroids_y[:, 2])
        return valid

    @torch.no_grad()
    def sample(self, base_conditions, base_region_masks, count=None):
        """Return fresh hierarchical conditions and exclusive style regions."""
        if base_conditions.shape[1] != 7:
            raise ValueError(
                "Hierarchical sampler expects 7 condition channels "
                f"(got {base_conditions.shape[1]})."
            )
        if base_region_masks.shape[1] != 6:
            raise ValueError(
                "Hierarchical sampler expects 6 exclusive style regions "
                f"(got {base_region_masks.shape[1]})."
            )
        count = count or base_conditions.shape[0]
        reference_conditions = base_conditions[:1].repeat(count, 1, 1, 1)
        reference_regions = base_region_masks[:1].repeat(count, 1, 1, 1)
        height, width = reference_conditions.shape[2:]
        accepted_conditions, accepted_regions = [], []
        remaining, attempts = count, 0

        while remaining > 0 and attempts < self.max_attempts:
            attempts += 1
            source = base_conditions[:1].repeat(remaining, 1, 1, 1)
            grid = self._global_grid(
                remaining,
                height,
                width,
                source.device,
                source.dtype,
            )
            primitives = source[:, :self.INTERNAL_END]
            warped = F.grid_sample(
                primitives,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            conditions, regions = self._rebuild(
                warped[:, self.SUPPORT:self.SUPPORT + 1],
                warped[:, self.HIATUS:self.HIATUS + 1],
                warped[:, self.INTERNAL_START:self.INTERNAL_END],
            )
            valid = self._valid(conditions, reference_conditions[:remaining])
            if valid.any():
                accepted_conditions.append(conditions[valid])
                accepted_regions.append(regions[valid])
                remaining -= int(valid.sum().item())

        if remaining:
            accepted_conditions.append(reference_conditions[:remaining])
            accepted_regions.append(reference_regions[:remaining])
        return {
            "conditions": torch.cat(accepted_conditions, dim=0)[:count],
            "masks": torch.cat(accepted_regions, dim=0)[:count],
        }
