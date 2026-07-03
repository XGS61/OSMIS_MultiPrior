import torch
import torch.nn.functional as F


class losses_computer():
    def __init__(self, opt, num_blocks):
        """
        The class implementing the loss computations
        """
        self.loss_function = self.get_loss_function(opt.loss_mode)
        self.no_masks = opt.no_masks
        self.no_DR = opt.no_DR
        self.lambdas = {"content": 0.5 / num_blocks,
                        "layout": 0.5 / num_blocks,
                        "low-level": 1.0 / num_blocks,
                        "DR": opt.lambda_DR}

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

    def diversity_regularization(self, fake):
        """
        The diversity regularization applied in the feature space of the generator
        """
        loss = torch.nn.L1Loss()
        ans = fake[0].new_tensor(0.0)
        for i in range(len(fake)):
            for k in range(fake[i].shape[0]):
                for m in range(k + 1, fake[i].shape[0]):
                    ans += -loss(fake[i][k], fake[i][m])
        pair_count = sum(
            item.shape[0] * (item.shape[0] - 1) // 2 for item in fake
        )
        return ans / max(pair_count, 1)

    def balance_losses(self, losses):
        """
        Multiply each loss part with its lambda
        """
        for item in losses:
            if item in self.lambdas.keys():
                losses[item] = losses[item] * self.lambdas[item]
        return losses

    def __call__(self, out_d, data, real, forD):
        losses = {}
        # --- adversarial loss ---#
        for item in out_d:
            for i in range(len(out_d[item])):
                if item == "content" and not self.no_masks:
                    losses[item] = losses.get(item, 0) + self.content_segm_loss(
                        out_d[item][i], data, real, forD
                    )
                else:
                    losses[item] = losses.get(item, 0) + self.loss_function(
                        out_d[item][i], real, forD
                    )

        # --- diversity regularization ---#
        if not forD and not self.no_DR:
            losses["DR"] = self.diversity_regularization(data["features"])
        losses = self.balance_losses(losses)
        return losses


def _masked_stats(image, mask, eps=1e-6):
    weights = mask.expand(-1, image.shape[1], -1, -1)
    count = weights.sum(dim=(2, 3)).clamp_min(eps)
    mean = (image * weights).sum(dim=(2, 3)) / count
    variance = (((image - mean[:, :, None, None]) ** 2) * weights).sum(dim=(2, 3)) / count
    return mean, torch.sqrt(variance + eps)


def _texture_statistics_loss(fake, real, fake_masks, real_masks):
    total = fake.new_tensor(0.0)
    for channel in range(fake_masks.shape[1]):
        fake_region = fake_masks[:, channel:channel + 1]
        real_region = real_masks[:, channel:channel + 1]
        fake_mean, fake_std = _masked_stats(fake, fake_region)
        real_mean, real_std = _masked_stats(real, real_region)
        total = total + F.l1_loss(fake_mean, real_mean) + F.l1_loss(fake_std, real_std)

    fake_gray = fake.mean(dim=1, keepdim=True)
    real_gray = real.mean(dim=1, keepdim=True)
    fake_grad = torch.abs(fake_gray[:, :, :, 1:] - fake_gray[:, :, :, :-1])
    real_grad = torch.abs(real_gray[:, :, :, 1:] - real_gray[:, :, :, :-1])
    fake_grad_mask = fake_masks[:, 1:2, :, 1:]
    real_grad_mask = real_masks[:, 1:2, :, 1:]
    fake_g_mean, fake_g_std = _masked_stats(fake_grad, fake_grad_mask)
    real_g_mean, real_g_std = _masked_stats(real_grad, real_grad_mask)
    return total + F.l1_loss(fake_g_mean, real_g_mean) + F.l1_loss(fake_g_std, real_g_std)


def _frequency_distribution_loss(fake, real):
    """Match multi-band frequency energy without forcing pixel correspondence."""
    fake_gray = F.interpolate(fake.mean(1, keepdim=True), size=(64, 64), mode="area")
    real_gray = F.interpolate(real.mean(1, keepdim=True), size=(64, 64), mode="area")
    total = fake.new_tensor(0.0)
    for kernel in (3, 7, 15):
        fake_low = F.avg_pool2d(
            fake_gray, kernel, stride=1, padding=kernel // 2
        )
        real_low = F.avg_pool2d(
            real_gray, kernel, stride=1, padding=kernel // 2
        )
        fake_band = fake_gray - fake_low
        real_band = real_gray - real_low
        total = total + F.l1_loss(
            fake_band.abs().mean((2, 3)), real_band.abs().mean((2, 3))
        )
        total = total + F.l1_loss(
            fake_band.std((2, 3)), real_band.std((2, 3))
        )
    return total / 3.0


def _rings(foreground, width=5):
    dilated = F.max_pool2d(foreground, width * 2 + 1, stride=1, padding=width)
    eroded = -F.max_pool2d(-foreground, width * 2 + 1, stride=1, padding=width)
    inner = (foreground - eroded).clamp(0, 1)
    outer = (dilated - foreground).clamp(0, 1)
    boundary = (inner + outer).clamp(0, 1)
    return inner, outer, boundary


def _boundary_signature(image, masks):
    foreground = masks[:, 1:2]
    inner, outer, boundary = _rings(foreground)
    gray = image.mean(1, keepdim=True)
    inner_mean, _ = _masked_stats(gray, inner)
    outer_mean, _ = _masked_stats(gray, outer)
    dx = F.pad(torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1]), (0, 1, 0, 0))
    dy = F.pad(torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :]), (0, 0, 0, 1))
    gradient_mean, gradient_std = _masked_stats(dx + dy, boundary)
    return outer_mean - inner_mean, gradient_mean, gradient_std


def _alignment_loss(fake, real, fake_masks, real_masks):
    fake_signature = _boundary_signature(fake, fake_masks)
    real_signature = _boundary_signature(real, real_masks)
    return sum(F.l1_loss(a, b) for a, b in zip(fake_signature, real_signature))


def _same_mask_diversity(fake_a, fake_b, masks, margin):
    image_a, image_b = fake_a["images"][-1], fake_b["images"][-1]
    inside = masks[:, 1:2]
    outside = masks[:, :1]
    inside_delta = ((image_a - image_b).abs() * inside).sum() / (
        inside.sum().clamp_min(1.0) * image_a.shape[1]
    )
    outside_delta = ((image_a - image_b).abs() * outside).sum() / (
        outside.sum().clamp_min(1.0) * image_a.shape[1]
    )
    return F.relu(image_a.new_tensor(margin) - 0.5 * (inside_delta + outside_delta))


def guided_generator_losses(fake_a, fake_b, anchor, real, opt):
    """Distributional constraints for one real image and mask-only conditions."""
    fake_image = fake_a["images"][-1]
    real_image = real["images"][-1]
    fake_masks = fake_a["masks"]
    real_masks = real["masks"]
    return {
        "texture": _texture_statistics_loss(
            fake_image, real_image, fake_masks, real_masks
        ) * opt.lambda_texture,
        "frequency": _frequency_distribution_loss(
            fake_image, real_image
        ) * opt.lambda_frequency,
        "alignment": _alignment_loss(
            fake_image, real_image, fake_masks, real_masks
        ) * opt.lambda_alignment,
        "same_mask_div": _same_mask_diversity(
            fake_a, fake_b, fake_masks, opt.diversity_margin
        ) * opt.lambda_same_mask_div,
        # Only this fixed latent/mask pair reconstructs the source image.
        "anchor": F.l1_loss(anchor["images"][-1], real_image[:1]) * opt.lambda_anchor,
    }


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
