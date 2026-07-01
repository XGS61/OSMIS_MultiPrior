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
        ans = 0
        for i in range(len(fake)):
            for k in range(fake[i].shape[0]):
                for m in range(k + 1, fake[i].shape[0]):
                    ans += -loss(fake[i][k], fake[i][m])
        return ans * 2 / (len(fake) * (len(fake) - 1))

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


def _soft_dice(prediction, target, eps=1e-6):
    prediction = prediction[:, 1:] if prediction.shape[1] > 1 else prediction
    target = target[:, 1:] if target.shape[1] > 1 else target
    intersection = torch.sum(prediction * target, dim=(1, 2, 3))
    denominator = torch.sum(prediction + target, dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + eps) / (denominator + eps)).mean()


def _boundary_loss(prediction, target):
    prediction = prediction[:, 1:2] if prediction.shape[1] > 1 else prediction[:, :1]
    target = target[:, 1:2] if target.shape[1] > 1 else target[:, :1]

    def gradients(tensor):
        dx = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
        dy = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
        return dx, dy

    pred_dx, pred_dy = gradients(prediction)
    target_dx, target_dy = gradients(target)
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def _low_frequency_loss(fake, real):
    kernel = max(4, min(fake.shape[2:]) // 20)
    fake_low = F.avg_pool2d(fake, kernel_size=kernel, stride=kernel)
    real_low = F.avg_pool2d(real, kernel_size=kernel, stride=kernel)
    return F.l1_loss(fake_low, real_low)


def _masked_stats(image, mask, eps=1e-6):
    weights = mask.expand(-1, image.shape[1], -1, -1)
    count = weights.sum(dim=(2, 3)).clamp_min(eps)
    mean = (image * weights).sum(dim=(2, 3)) / count
    variance = (((image - mean[:, :, None, None]) ** 2) * weights).sum(dim=(2, 3)) / count
    return mean, torch.sqrt(variance + eps)


def _texture_statistics_loss(fake, real, masks):
    total = fake.new_tensor(0.0)
    for channel in range(masks.shape[1]):
        region = masks[:, channel:channel + 1]
        fake_mean, fake_std = _masked_stats(fake, region)
        real_mean, real_std = _masked_stats(real, region)
        total = total + F.l1_loss(fake_mean, real_mean) + F.l1_loss(fake_std, real_std)

    fake_gray = fake.mean(dim=1, keepdim=True)
    real_gray = real.mean(dim=1, keepdim=True)
    fake_grad = torch.abs(fake_gray[:, :, :, 1:] - fake_gray[:, :, :, :-1])
    real_grad = torch.abs(real_gray[:, :, :, 1:] - real_gray[:, :, :, :-1])
    grad_mask = masks[:, 1:2, :, 1:] if masks.shape[1] > 1 else masks[:, :1, :, 1:]
    fake_g_mean, fake_g_std = _masked_stats(fake_grad, grad_mask)
    real_g_mean, real_g_std = _masked_stats(real_grad, grad_mask)
    return total + F.l1_loss(fake_g_mean, real_g_mean) + F.l1_loss(fake_g_std, real_g_std)


def guided_generator_losses(fake, real, opt):
    """Losses that make the external target mask control anatomy."""
    prediction = fake["pred_masks"]
    target = fake["masks"]
    fake_image = fake["images"][-1]
    real_image = real["images"][-1]
    return {
        "seg": _soft_dice(prediction, target) * opt.lambda_seg,
        "boundary": _boundary_loss(prediction, target) * opt.lambda_boundary,
        "lowfreq": _low_frequency_loss(fake_image, real_image) * opt.lambda_lowfreq,
        "texture": _texture_statistics_loss(fake_image, real_image, target) * opt.lambda_texture,
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
