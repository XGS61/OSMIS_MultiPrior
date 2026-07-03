import config
from core import dataloading, models, utils, tracking
from core.mask_prior import MaskPriorBank


# --- read options --- #
opt = config.read_arguments(train=False)

# --- create dataloader and recommended model config --- #
dataloader, model_config = dataloading.prepare_dataloading(opt)

# --- create models, losses, and optimizers ---#
netG, netD, netEMA = models.create_models(opt, model_config)

# --- create utils --- #
visualizer = tracking.visualizer(opt)
mask_prior_dir = opt.mask_prior_dir or (
    f"{opt.dataroot}/{opt.dataset_name}/mask_priors"
)
mask_bank = MaskPriorBank(mask_prior_dir, model_config["image resolution"], opt.device)

# --- generate images and masks --- #
data_iterator = iter(dataloader)
for i in range(opt.num_generated):
    batch = next(data_iterator)
    batch = utils.preprocess_real(batch, model_config["num_blocks_d0"], opt.device)
    target_mask = mask_bank.cycle(1, offset=i)
    z = utils.sample_noise(opt.noise_dim, 1).to(opt.device)
    fake = (
        netEMA.generate(
            z,
            masks=target_mask,
            style_images=batch["images"][-1][:1],
            style_masks=batch["masks"][:1],
        )
        if not opt.no_EMA
        else netG.generate(
            z,
            masks=target_mask,
            style_images=batch["images"][-1][:1],
            style_masks=batch["masks"][:1],
        )
    )
    visualizer.save_batch(fake, opt.continue_epoch, i=str(i))
