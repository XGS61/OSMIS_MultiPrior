import torch
import torch.nn.functional as F


class losses_computer():
    def __init__(self, opt, num_blocks):
        """
        The class implementing the loss computations
        """
        self.loss_function = self.get_loss_function(opt.loss_mode)
        self.no_masks = opt.no_masks
        self.num_blocks = num_blocks
        self.lambda_content = opt.lambda_content
        self.lambda_layout = opt.lambda_layout
        self.layout_decay_start = opt.layout_decay_start
        self.layout_decay_end = opt.layout_decay_end
        self.layout_final_ratio = opt.layout_final_ratio

    def get_loss_function(self, loss_mode):
        if loss_mode == "wgan":
            return wgan_loss
        elif loss_mode == "hinge":
            return hinge_loss
        elif loss_mode == "bce":
            return bce_loss
        else:
            raise ValueError('Unexpected loss_mode {}'.format(mode))

    def content_segm_loss(self, out_d, data, real, forD):
        """
        The multi-class cross-entropy loss used in the content masked attention
        """
        mask = data["masks"]
        mask_ch = mask.shape[1]
        if real:
            ground_t = torch.arange(mask_ch).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            ground_t = ground_t.repeat(1, 1, out_d.shape[2], out_d.shape[3])
            ground_t = ground_t.repeat_interleave(mask.shape[0], dim=0)[:, 0, :, :]
        else:  # fake
            ground_t = torch.ones_like(out_d)[:, 0, :, :] * mask_ch
        weights = torch.cat((1 / (torch.sum(mask.detach(), dim=(0, 2, 3))), torch.Tensor([1.0]).to(out_d.device)))
        weights[weights == float('inf')] = 0
        loss = F.cross_entropy(out_d, ground_t.long().to(out_d.device), weight=weights.to(out_d.device))
        return loss

    def layout_weight(self, epoch):
        if epoch <= self.layout_decay_start:
            ratio = 1.0
        elif epoch >= self.layout_decay_end:
            ratio = self.layout_final_ratio
        else:
            progress = (epoch - self.layout_decay_start) / max(
                self.layout_decay_end - self.layout_decay_start, 1
            )
            ratio = 1.0 - progress * (1.0 - self.layout_final_ratio)
        return self.lambda_layout * ratio / self.num_blocks

    def __call__(self, out_d, data, real, forD, epoch=0):
        losses = {}
        # --- adversarial loss ---#
        for item in ("low-level", "content", "layout"):
            for i in range(len(out_d[item])):
                if item == "content" and not self.no_masks:
                    losses[item] = losses.get(item, 0) + self.content_segm_loss(
                        out_d[item][i], data, real, forD
                    )
                else:
                    losses[item] = losses.get(item, 0) + self.loss_function(
                        out_d[item][i], real, forD
                    )

        losses["low-level"] = losses["low-level"] / self.num_blocks
        losses["content"] = (
            losses["content"] * self.lambda_content / self.num_blocks
        )
        losses["layout"] = losses["layout"] * self.layout_weight(epoch)
        return losses


def _masked_stats(image, mask, eps=1e-6):
    weights = mask.expand(-1, image.shape[1], -1, -1)
    count = weights.sum(dim=(2, 3)).clamp_min(eps)
    mean = (image * weights).sum(dim=(2, 3)) / count
    variance = (((image - mean[:, :, None, None]) ** 2) * weights).sum(dim=(2, 3)) / count
    return mean, torch.sqrt(variance + eps)


def _rings(foreground, width=5):
    dilated = F.max_pool2d(foreground, width * 2 + 1, stride=1, padding=width)
    eroded = -F.max_pool2d(-foreground, width * 2 + 1, stride=1, padding=width)
    inner = (foreground - eroded).clamp(0, 1)
    outer = (dilated - foreground).clamp(0, 1)
    boundary = (inner + outer).clamp(0, 1)
    return inner, outer, boundary


def _boundary_signature(image, foreground):
    inner, outer, boundary = _rings(foreground)
    gray = image.mean(1, keepdim=True)
    inner_mean, _ = _masked_stats(gray, inner)
    outer_mean, _ = _masked_stats(gray, outer)
    dx = F.pad(torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1]), (0, 1, 0, 0))
    dy = F.pad(torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :]), (0, 0, 0, 1))
    gradient_mean, gradient_std = _masked_stats(dx + dy, boundary)
    return outer_mean - inner_mean, gradient_mean, gradient_std


def _structure_consistency(fake, real, fake_masks, real_masks):
    """One orthogonal structural objective for every annotated foreground class."""
    total = fake.new_tensor(0.0)
    class_count = max(fake_masks.shape[1] - 1, 1)
    for channel in range(1, fake_masks.shape[1]):
        fake_signature = _boundary_signature(
            fake, fake_masks[:, channel:channel + 1]
        )
        real_signature = _boundary_signature(
            real, real_masks[:, channel:channel + 1]
        )
        total = total + sum(
            F.l1_loss(a, b) for a, b in zip(fake_signature, real_signature)
        )
    return total / class_count


def _decayed_weight(initial, final_ratio, start, end, epoch):
    if epoch <= start:
        ratio = 1.0
    elif epoch >= end:
        ratio = final_ratio
    else:
        progress = (epoch - start) / max(end - start, 1)
        ratio = 1.0 - progress * (1.0 - final_ratio)
    return initial * ratio


def guided_generator_losses(fake, anchor, real, opt, epoch):
    """Minimal non-adversarial objectives: structure plus one reconstruction anchor."""
    fake_image = fake["images"][-1]
    real_image = real["images"][-1]
    fake_masks = fake["masks"]
    real_masks = real["masks"]
    anchor_weight = _decayed_weight(
        opt.lambda_anchor,
        opt.anchor_final_ratio,
        opt.anchor_decay_start,
        opt.anchor_decay_end,
        epoch,
    )
    return {
        "structure": _structure_consistency(
            fake_image, real_image, fake_masks, real_masks
        ) * opt.lambda_structure,
        "anchor": F.l1_loss(
            anchor["images"][-1], real_image[:1]
        ) * anchor_weight,
    }


def texture_diversity_loss(fake_a, fake_b, masks, z_a, z_b, weight, cap):
    """Make region texture latents visible without rewarding anatomy changes."""
    image_a = fake_a["images"][-1]
    image_b = fake_b["images"][-1]
    high_a = image_a - F.avg_pool2d(image_a, 7, stride=1, padding=3)
    high_b = image_b - F.avg_pool2d(image_b, 7, stride=1, padding=3)
    difference = (high_a - high_b).abs().mean(dim=1, keepdim=True)

    region_masks = masks[:, 1:]
    area = region_masks.sum((2, 3)).clamp_min(1.0)
    image_distance = (
        difference * region_masks
    ).sum((2, 3)) / area
    latent_distance = (z_a[:, 1:] - z_b[:, 1:]).abs().mean(dim=2)
    ratio = image_distance / latent_distance.clamp_min(1e-4)
    valid = (area >= 16).to(ratio.dtype)
    score = (torch.clamp(ratio, max=cap) * valid).sum() / valid.sum().clamp_min(1)
    return -score * weight


def wgan_loss(output, real, forD):
    if real and forD:
        ans = -output.mean()
    elif not real and forD:
        ans = output.mean()
    elif real and not forD:
        ans = -output.mean()
    elif not real and not forD:
        raise ValueError("gen loss should be for real")
    #print(real, forD, ans)
    return ans


def hinge_loss(output, real, forD):
    if real and forD:
        minval = torch.min(output - 1, get_zero_tensor(output).to(output.device))
        ans = -torch.mean(minval)
    elif not real and forD:
        minval = torch.min(-output - 1, get_zero_tensor(output).to(output.device))
        ans = -torch.mean(minval)
    elif real and not forD:
        ans = -torch.mean(output)
    elif not real and not forD:
        raise ValueError("gen loss should be for real")
    return ans


def bce_loss(output, real, forD, no_aggr=False):
    target_tensor = get_target_tensor(output, real).to(output.device)
    ans = F.binary_cross_entropy_with_logits(output, target_tensor, reduction=("mean" if not no_aggr else "none"))
    return ans


def get_target_tensor(input, target_is_real):
    if target_is_real:
        real_label_tensor = torch.FloatTensor(1).fill_(1)
        real_label_tensor.requires_grad_(False)
    else:
        real_label_tensor = torch.FloatTensor(1).fill_(0)
        real_label_tensor.requires_grad_(False)
    return real_label_tensor.expand_as(input)


def get_zero_tensor(input):
    zero_tensor = torch.FloatTensor(1).fill_(0)
    zero_tensor.requires_grad_(False)
    return zero_tensor.expand_as(input)
