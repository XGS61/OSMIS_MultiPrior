"""Mask-only condition bank for one-shot training."""

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


class MaskPriorBank:
    def __init__(self, root, target_size, device):
        self.paths = sorted(Path(root).glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"No mask priors found in {root}")
        masks = []
        for path in self.paths:
            tensor = pil_to_tensor(Image.open(path).convert("L")).float() / 255.0
            tensor = (tensor >= 0.5).float()
            masks.append(tensor)
        stacked = torch.stack(masks)
        stacked = F.interpolate(stacked, size=target_size, mode="nearest")
        self.masks = torch.cat((1.0 - stacked, stacked), dim=1).to(device)

    def __len__(self):
        return self.masks.shape[0]

    def sample(self, batch_size):
        indices = torch.randint(0, len(self), (batch_size,), device=self.masks.device)
        return self.masks[indices]

    def first(self, batch_size=1):
        return self.masks[:1].repeat(batch_size, 1, 1, 1)

    def cycle(self, batch_size, offset=0):
        indices = (
            torch.arange(batch_size, device=self.masks.device) + offset
        ) % len(self)
        return self.masks[indices]
