import config
from core import dataloading, models, tracking, utils
from core.online_anatomy import OnlineAnatomySampler


opt = config.read_arguments(train=False)
dataloader, model_config = dataloading.prepare_dataloading(opt)
netG, netD, netEMA = models.create_models(opt, model_config)
visualizer = tracking.visualizer(opt)
anatomy_sampler = OnlineAnatomySampler(
    max_displacement_frac=opt.anatomy_max_displacement,
    support_max_displacement_frac=opt.support_max_displacement,
    support_max_rotation_deg=opt.support_max_rotation,
    support_max_translation_frac=opt.support_max_translation,
)

data_iterator = iter(dataloader)
for index in range(opt.num_generated):
    batch = next(data_iterator)
    batch = utils.preprocess_real(
        batch, model_config["num_blocks_d0"], opt.device
    )
    base_mask = batch["masks"][:1]
    target_mask = anatomy_sampler.sample(base_mask, count=1)
    z_global = utils.sample_noise(opt.global_noise_dim, 1).to(opt.device)
    z_texture = utils.sample_noise(opt.texture_noise_dim, 1).to(opt.device)
    generator = netEMA if not opt.no_EMA else netG
    fake = generator.generate(
        z_global,
        z_texture,
        masks=target_mask,
        style_images=batch["images"][-1][:1],
        style_masks=base_mask,
    )
    visualizer.save_batch(fake, opt.continue_epoch, i=str(index))
