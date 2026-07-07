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
print("Using fresh online anatomy conditions; no fixed mask bank is loaded.")


def clone_batch(data):
    return {
        key: [value.clone() for value in item]
        if isinstance(item, list)
        else item.clone()
        for key, item in data.items()
    }


def sample_latents(count):
    z_global = utils.sample_noise(opt.global_noise_dim, count).to(opt.device)
    z_texture = utils.sample_noise(opt.texture_noise_dim, count).to(opt.device)
    return z_global, z_texture


for epoch, batch in enumerate(dataloader, start=opt.continue_epoch):
    batch = utils.preprocess_real(batch, netD.num_blocks_ll, opt.device)
    style_images = batch["images"][-1]
    style_masks = batch["masks"]
    batch_size = style_images.shape[0]
    logits, losses = {}, {}

    # ---------------- Generator update ----------------
    netG.zero_grad()
    target_masks = anatomy_sampler.sample(style_masks, count=batch_size)
    z_global, z_texture = sample_latents(batch_size)
    out_G = netG.generate(
        z_global,
        z_texture,
        masks=target_masks,
        style_images=style_images,
        style_masks=style_masks,
    )
    anchor = netG.generate(
        torch.zeros(
            1, opt.global_noise_dim, 1, 1, device=opt.device
        ),
        torch.zeros(
            1, opt.texture_noise_dim, 1, 1, device=opt.device
        ),
        masks=style_masks[:1],
        style_images=style_images[:1],
        style_masks=style_masks[:1],
        randomize_noise=False,
    )
    losses["G"] = losses_module.guided_generator_losses(
        out_G, anchor, batch, opt, epoch
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
    losses["G"]["latent"] = losses_module.latent_reconstruction_loss(
        logits["G"]["latent"], z_texture, opt.lambda_latent
    )
    sum(losses["G"].values()).backward()
    optimizerG.step()

    # ---------------- Discriminator + latent head update ----------------
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

    target_masks = anatomy_sampler.sample(style_masks, count=batch_size)
    z_global_d, z_texture_d = sample_latents(batch_size)
    with torch.no_grad():
        out_G_d = netG.generate(
            z_global_d,
            z_texture_d,
            masks=target_masks,
            style_images=style_images,
            style_masks=style_masks,
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
    # E_z shares D features and is trained only on known synthetic latents.
    losses["Dfake"]["latent"] = losses_module.latent_reconstruction_loss(
        logits["Dfake"]["latent"], z_texture_d, opt.lambda_latent
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
        repeated_images = style_images[:1].repeat(count, 1, 1, 1)
        repeated_masks = style_masks[:1].repeat(count, 1, 1, 1)

        # First row group: same anatomy/global code, different texture latents.
        same_condition = anatomy_sampler.sample(style_masks[:1], count=1).repeat(
            count, 1, 1, 1
        )
        fixed_global, _ = sample_latents(1)
        fixed_global = fixed_global.repeat(count, 1, 1, 1)
        _, varied_texture = sample_latents(count)
        fake_texture = monitor_net.generate(
            fixed_global,
            varied_texture,
            masks=same_condition,
            style_images=repeated_images,
            style_masks=repeated_masks,
            randomize_noise=False,
        )

        # Second row group: same latents, fresh online anatomy conditions.
        varied_conditions = anatomy_sampler.sample(style_masks[:1], count=count)
        fixed_global_2, fixed_texture_2 = sample_latents(1)
        fake_anatomy = monitor_net.generate(
            fixed_global_2.repeat(count, 1, 1, 1),
            fixed_texture_2.repeat(count, 1, 1, 1),
            masks=varied_conditions,
            style_images=repeated_images,
            style_masks=repeated_masks,
            randomize_noise=False,
        )
        monitor = {
            "images": [
                torch.cat((left, right), dim=0)
                for left, right in zip(
                    fake_texture["images"], fake_anatomy["images"]
                )
            ],
            "masks": torch.cat((same_condition, varied_conditions), dim=0),
        }
        visualizer.save_batch(monitor, epoch)
    if (
        epoch % opt.freq_save_loss == 0 or epoch == opt.num_epochs
    ) and epoch > 0:
        visualizer.save_losses_logits(epoch)
    if epoch >= opt.num_epochs:
        break

print("Successfully finished")
