import config
from core import dataloading, models, utils, losses as losses_module, tracking
from core.mask_prior import MaskPriorBank
from core.differentiable_augmentation import diff_augm
import torch


# --- read options --- #
opt = config.read_arguments(train=True)

# --- create dataloader and recommended model config --- #
dataloader, model_config = dataloading.prepare_dataloading(opt)

# --- create models, losses, and optimizers ---#
netG, netD, netEMA = models.create_models(opt, model_config)
losses_computer = losses_module.losses_computer(opt, netD.num_blocks)
optimizerG, optimizerD = models.create_optimizers(netG, netD, opt)

# --- create utils --- #
utils.fix_seed(opt.seed)
timer = utils.timer(opt)
visualizer = tracking.visualizer(opt)
diff_augment = diff_augm.augment_pipe(opt)
mask_prior_dir = opt.mask_prior_dir or (
    f"{opt.dataroot}/{opt.dataset_name}/mask_priors"
)
mask_bank = MaskPriorBank(mask_prior_dir, model_config["image resolution"], opt.device)
print(f"Loaded {len(mask_bank)} mask-only anatomy priors from {mask_prior_dir}")


def clone_batch(data):
    return {
        key: [value.clone() for value in item] if isinstance(item, list) else item.clone()
        for key, item in data.items()
    }

# --- training loop --- #
for epoch, batch in enumerate(dataloader, start=opt.continue_epoch):
    batch = utils.preprocess_real(batch, netD.num_blocks_ll, opt.device)
    style_images = batch["images"][-1]
    style_masks = batch["masks"]
    logits, losses = dict(), dict()

    # --- generator update --- #
    netG.zero_grad()
    target_masks = mask_bank.sample(style_images.shape[0])
    z = utils.sample_noise(opt.noise_dim, style_images.shape[0]).to(opt.device)
    z_second = utils.sample_noise(opt.noise_dim, style_images.shape[0]).to(opt.device)
    out_G = netG.generate(
        z,
        masks=target_masks,
        style_images=style_images,
        style_masks=style_masks,
        get_feat=not opt.no_DR,
    )
    out_G_second = netG.generate(
        z_second,
        masks=target_masks,
        style_images=style_images,
        style_masks=style_masks,
    )
    anchor = netG.generate(
        torch.zeros(1, opt.noise_dim, 1, 1, device=opt.device),
        masks=style_masks[:1],
        style_images=style_images[:1],
        style_masks=style_masks[:1],
        randomize_noise=False,
    )
    losses["G"] = losses_module.guided_generator_losses(
        out_G, out_G_second, anchor, batch, opt
    )
    out_G = diff_augment(out_G)
    logits["G"] = netD.discriminate(out_G, for_real=False, epoch=epoch)
    losses["G"].update(losses_computer(logits["G"], out_G, real=True, forD=False))
    loss = sum(losses["G"].values())
    loss.backward()
    optimizerG.step()

    # --- discriminator update --- #
    netD.zero_grad()
    batch_augmented = diff_augment(clone_batch(batch))
    logits["Dreal"] = netD.discriminate(batch_augmented, for_real=True, epoch=epoch)
    losses["Dreal"] = losses_computer(
        logits["Dreal"], batch_augmented, real=True, forD=True
    )
    loss = sum(losses["Dreal"].values())
    loss.backward()

    z = utils.sample_noise(opt.noise_dim, style_images.shape[0]).to(opt.device)
    target_masks = mask_bank.sample(style_images.shape[0])
    with torch.no_grad():
        out_G = netG.generate(
            z,
            masks=target_masks,
            style_images=style_images,
            style_masks=style_masks,
        )
    out_G = diff_augment(out_G)
    logits["Dfake"] = netD.discriminate(out_G, for_real=False, epoch=epoch)
    losses["Dfake"] = losses_computer(logits["Dfake"], out_G, real=False, forD=True)
    loss = sum(losses["Dfake"].values())
    loss.backward()
    optimizerD.step()

    # --- stats tracking --- #
    visualizer.track_losses_logits(logits, losses)
    if not opt.no_EMA:
        netEMA = utils.update_EMA(netEMA, netG, opt.EMA_decay)
    if epoch % opt.freq_save_ckpt == 0 or epoch == opt.num_epochs:
        visualizer.save_networks(netG, netD, netEMA, epoch)
    if epoch % opt.freq_print == 0 or epoch == opt.num_epochs:
        timer(epoch)
        monitor_net = netEMA if not opt.no_EMA else netG
        same_masks = mask_bank.first(8)
        varied_masks = mask_bank.cycle(8, offset=epoch)
        z_random = utils.sample_noise(opt.noise_dim, 8).to(opt.device)
        z_fixed = utils.sample_noise(opt.noise_dim, 1).to(opt.device).repeat(8, 1, 1, 1)
        repeated_style_images = style_images[:1].repeat(8, 1, 1, 1)
        repeated_style_masks = style_masks[:1].repeat(8, 1, 1, 1)
        fake_same = monitor_net.generate(
            z_random,
            masks=same_masks,
            style_images=repeated_style_images,
            style_masks=repeated_style_masks,
        )
        fake_varied = monitor_net.generate(
            z_fixed,
            masks=varied_masks,
            style_images=repeated_style_images,
            style_masks=repeated_style_masks,
        )
        fake = {
            "images": [
                torch.cat((left, right), dim=0)
                for left, right in zip(fake_same["images"], fake_varied["images"])
            ],
            "masks": torch.cat((same_masks, varied_masks), dim=0),
        }
        visualizer.save_batch(fake, epoch)
    if (epoch % opt.freq_save_loss == 0 or epoch == opt.num_epochs) and epoch > 0 :
        visualizer.save_losses_logits(epoch)

    # --- exit if reached the end --- #
    if epoch >= opt.num_epochs:
        break

# --- after training ---#
print("Succesfully finished")
