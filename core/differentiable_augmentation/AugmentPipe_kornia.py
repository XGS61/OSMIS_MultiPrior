import torch
import random
from torchvision import transforms as TR
import torch.nn.functional as F


class AugmentPipe_kornia(torch.nn.Module):
    def __init__(self, prob, no_masks):
        super().__init__()
        self.prob = prob
        self.no_masks = no_masks

    def forward(self, batch):
        # Geometry is supplied by the validated pseudo-pair generator.  The
        # original OSMIS crops, translations and object copy/paste operations
        # are intentionally excluded because they can invalidate C-plane
        # anatomy.  Only mild differentiable photometric jitter is applied.
        images = batch["images"]
        if random.random() >= self.prob:
            return batch

        batch_size = images[-1].shape[0]
        device = images[-1].device
        dtype = images[-1].dtype
        brightness = (torch.rand(batch_size, 1, 1, 1, device=device, dtype=dtype) - 0.5) * 0.10
        contrast = 0.95 + torch.rand(batch_size, 1, 1, 1, device=device, dtype=dtype) * 0.10
        noise_scale = torch.rand(batch_size, 1, 1, 1, device=device, dtype=dtype) * 0.012

        augmented = []
        for image in images:
            noise = torch.randn_like(image) * noise_scale
            augmented.append(torch.clamp((image + brightness) * contrast + noise, -1.0, 1.0))
        batch["images"] = augmented
        return batch


def combine_fakes(inp):
    sh = inp[-1].shape
    ans = list()
    for i in range(sh[0]):
        cur = torch.zeros_like(inp[-1][0, :, :, :]).repeat(len(inp), 1, 1, 1)
        for j in range(len(inp)):
            cur[j, :, :, :] = F.interpolate(inp[j][i, :, :, :].unsqueeze(0), size=(sh[2], sh[3]),
                                                              mode="bilinear")
        ans.append(cur)
    return ans


def detach_fakes(inp, ref):
    ans = list()
    sh = ref[-1].shape
    for i in range(len(ref)):
        cur = torch.zeros_like(ref[i])
        for j in range(sh[0]):
            cur[j, :, :, :] = F.interpolate(inp[j][i, :, :, :].unsqueeze(0),
                                                              size=(ref[i].shape[2], ref[i].shape[3]),
                                                              mode="bilinear")
        ans.append(cur)
    return ans


class myRandomResizedCrop(TR.RandomResizedCrop):
    def __init__(self, size=256, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), ):
        super(myRandomResizedCrop, self).__init__(size, scale, ratio)

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return TR.functional.resized_crop(img, i, j, h, w, (img.size[1], img.size[0]), self.interpolation)


def translate_v_fake(x, fraction):
    margin = torch.rand(1) * (fraction[1] - fraction[0]) + fraction[0]
    direct_up = (torch.rand(1) < 0.5)  # up or down
    height, width = x.shape[2], x.shape[3]
    left, right = 0, width
    if direct_up:
        top, bottom = 0, int(height * margin)
    else:
        top, bottom = height - int(height * margin), height
    im_to_paste = torch.flip(x[:, :, top:bottom, left:right], (2,))
    if not direct_up:
        x[:, :, 0:height - int(height * margin), :] = x[:, :, int(height * margin):height, :].clone()
        x[:, :, height - int(height * margin):, :] = im_to_paste
    else:
        x[:, :, int(height * margin):height, :] = x[:, :, 0:height - int(height * margin), :].clone()
        x[:, :, :int(height * margin), :] = im_to_paste
    return x


def translate_h_fake(x, fraction):
    margin = torch.rand(1) * (fraction[1] - fraction[0]) + fraction[0]
    direct_left = (torch.rand(1) < 0.5)  # up or down
    height, width = x.shape[2], x.shape[3]
    top, bottom = 0, height
    if direct_left:
        left, right = 0, int(width * margin)
    else:
        left, right = width - int(width * margin), width
    im_to_paste = torch.flip(x[:, :, top:bottom, left:right], (3,))
    if not direct_left:
        x[:, :, :, 0:width - int(width * margin)] = x[:, :, :, int(width * margin):width].clone()
        x[:, :, :, width - int(width * margin):] = im_to_paste
    else:
        x[:, :, :, int(width * margin):width] = x[:, :, :, 0:width - int(width * margin)].clone()
        x[:, :, :, :int(width * margin)] = im_to_paste
    return x
