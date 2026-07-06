import torch

import config
from core import dataloading, losses as losses_module, models, tracking, utils
from core.differentiable_augmentation import diff_augm
from core.online_anatomy import OnlineAnatomySampler


opt = config.read_arguments(train=True)
dataloader, model_config = dataloading.prepare_dataloading(opt)
netG, netD, netEMA = models.create_models(opt, model_config)
adversarial_losses = losses_module.losses_computer(opt, netD.num_blocks)
optimizerG, optimizerD = models.create_optimizers(netG, netD, opt)

utils.fix_seed(opt.seed)
timer = utils.timer(opt)
visualizer = tracking.visualizer(opt)
diff_augment = diff_augm.augment_pipe(opt)
anatomy_sampler = OnlineAnatomySampler(
    max_displacement_frac=opt.anatomy_max_displacement
)
num_regions = model_config["num_mask_channels"]
print(
    "Using fresh hierarchical anatomy conditions; no fixed mask bank or "
    "pseudo-real image set is loaded."
)


def clone_batch(data):
    return {
        key: [value.clone() for value in item]
        if isinstance(item, list)
        else item.clone()
        for key, item in data.items()
    }


def sample_texture(count):
    return torch.randn(
        count,
        num_regions,
        opt.texture_noise_dim,
        device=opt.device,
    )


def mean_pairwise_l1(images):
    if images.shape[0] < 2:
        return images.new_tensor(0.0)
    distances = []
    for left in range(images.shape[0]):
        for right in range(left + 1, images.shape[0]):
            distances.append((images[left] - images[right]).abs().mean())
    return torch.stack(distances).mean()


for epoch, batch in enumerate(dataloader, start=opt.continue_epoch):
    batch = utils.preprocess_real(batch, netD.num_blocks_ll, opt.device)
    style_images = batch["images"][-1]
    style_masks = batch["masks"]
    base_conditions = batch["conditions"]
    batch_size = style_images.shape[0]
    logits, losses = {}, {}

    # ---------------- Generator update ----------------
    netG.zero_grad()
    sampled = anatomy_sampler.sample(
        base_conditions, style_masks, count=batch_size
    )
    style_codes = netG.encode_style(
        style_images[:1],
        style_masks[:1],
        output_count=batch_size,
        randomize_patches=True,
    )
    z_texture_a = sample_texture(batch_size)
    z_texture_b = sample_texture(batch_size)
    out_G = netG.generate(
        z_texture_a,
        conditions=sampled["conditions"],
        masks=sampled["masks"],
        style_codes=style_codes,
    )
    out_G_pair = netG.generate(
        z_texture_b,
        conditions=sampled["conditions"],
        masks=sampled["masks"],
        style_codes=style_codes,
    )
    anchor_style = netG.encode_style(
        style_images[:1],
        style_masks[:1],
        output_count=1,
        randomize_patches=False,
    )
    anchor = netG.generate(
        torch.zeros(
            1,
            num_regions,
            opt.texture_noise_dim,
            device=opt.device,
        ),
        conditions=base_conditions[:1],
        masks=style_masks[:1],
        style_codes=anchor_style,
    )
    losses["G"] = losses_module.guided_generator_losses(
        out_G, anchor, batch, opt, epoch
    )
    losses["G"]["texture_diversity"] = (
        losses_module.texture_diversity_loss(
            out_G,
            out_G_pair,
            sampled["masks"],
            z_texture_a,
            z_texture_b,
            opt.lambda_diversity,
            opt.diversity_cap,
        )
    )
    out_G_augmented = diff_augment(out_G)
    logits["G"] = netD.discriminate(
        out_G_augmented, for_real=False, epoch=epoch
    )
    losses["G"].update(
        adversarial_losses(
            logits["G"], out_G_augmented, real=True, forD=False, epoch=epoch
        )
    )
    sum(losses["G"].values()).backward()
    optimizerG.step()

    # ---------------- Discriminator update ----------------
    netD.zero_grad()
    batch_augmented = diff_augment(clone_batch(batch))
    logits["Dreal"] = netD.discriminate(
        batch_augmented, for_real=True, epoch=epoch
    )
    losses["Dreal"] = adversarial_losses(
        logits["Dreal"],
        batch_augmented,
        real=True,
        forD=True,
        epoch=epoch,
    )
    sum(losses["Dreal"].values()).backward()

    sampled_d = anatomy_sampler.sample(
        base_conditions, style_masks, count=batch_size
    )
    with torch.no_grad():
        style_codes_d = netG.encode_style(
            style_images[:1],
            style_masks[:1],
            output_count=batch_size,
            randomize_patches=True,
        )
        out_G_d = netG.generate(
            sample_texture(batch_size),
            conditions=sampled_d["conditions"],
            masks=sampled_d["masks"],
            style_codes=style_codes_d,
        )
    out_G_d = diff_augment(out_G_d)
    logits["Dfake"] = netD.discriminate(
        out_G_d, for_real=False, epoch=epoch
    )
    losses["Dfake"] = adversarial_losses(
        logits["Dfake"],
        out_G_d,
        real=False,
        forD=True,
        epoch=epoch,
    )
    sum(losses["Dfake"].values()).backward()
    optimizerD.step()

    # ---------------- Tracking ----------------
    visualizer.track_losses_logits(logits, losses)
    if not opt.no_EMA:
        netEMA = utils.update_EMA(netEMA, netG, opt.EMA_decay)
    if epoch % opt.freq_save_ckpt == 0 or epoch == opt.num_epochs:
        visualizer.save_networks(netG, netD, netEMA, epoch)
    if epoch % opt.freq_print == 0 or epoch == opt.num_epochs:
        timer(epoch)
        monitor_net = netEMA if not opt.no_EMA else netG
        count = 8
        monitor_style = monitor_net.encode_style(
            style_images[:1],
            style_masks[:1],
            output_count=count,
            randomize_patches=False,
        )

        # First group: identical anatomy and style reference, varied z_texture.
        one_condition = anatomy_sampler.sample(
            base_conditions[:1], style_masks[:1], count=1
        )
        same_conditions = one_condition["conditions"].repeat(
            count, 1, 1, 1
        )
        same_masks = one_condition["masks"].repeat(count, 1, 1, 1)
        fake_texture = monitor_net.generate(
            sample_texture(count),
            conditions=same_conditions,
            masks=same_masks,
            style_codes=monitor_style,
        )

        # Second group: fixed texture, fresh valid hierarchical anatomy.
        varied = anatomy_sampler.sample(
            base_conditions[:1], style_masks[:1], count=count
        )
        fixed_texture = sample_texture(1).repeat(count, 1, 1)
        fake_anatomy = monitor_net.generate(
            fixed_texture,
            conditions=varied["conditions"],
            masks=varied["masks"],
            style_codes=monitor_style,
        )
        monitor = {
            "images": [
                torch.cat((left, right), dim=0)
                for left, right in zip(
                    fake_texture["images"], fake_anatomy["images"]
                )
            ],
            "masks": torch.cat((same_masks, varied["masks"]), dim=0),
            "conditions": torch.cat(
                (same_conditions, varied["conditions"]), dim=0
            ),
        }
        visualizer.save_batch(monitor, epoch)
        texture_images = fake_texture["images"][-1]
        high_texture = texture_images - torch.nn.functional.avg_pool2d(
            texture_images, 7, stride=1, padding=3
        )
        low_texture = torch.nn.functional.avg_pool2d(
            texture_images, 7, stride=1, padding=3
        )
        monitor_message = (
            f"[monitor {epoch}] texture_high_l1="
            f"{float(mean_pairwise_l1(high_texture)):.6f}, "
            f"texture_low_l1={float(mean_pairwise_l1(low_texture)):.6f}, "
            f"anatomy_image_l1="
            f"{float(mean_pairwise_l1(fake_anatomy['images'][-1])):.6f}, "
            f"anatomy_mask_l1="
            f"{float(mean_pairwise_l1(varied['masks'])):.6f}"
        )
        print(monitor_message)
        with open(timer.file_name, "a", encoding="utf-8") as monitor_log:
            monitor_log.write(monitor_message + "\n")
    if (
        epoch % opt.freq_save_loss == 0 or epoch == opt.num_epochs
    ) and epoch > 0:
        visualizer.save_losses_logits(epoch)
    if epoch >= opt.num_epochs:
        break

print("Successfully finished")
