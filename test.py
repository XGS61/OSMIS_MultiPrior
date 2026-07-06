import torch

import config
from core import dataloading, models, tracking, utils
from core.online_anatomy import OnlineAnatomySampler


opt = config.read_arguments(train=False)
dataloader, model_config = dataloading.prepare_dataloading(opt)
netG, netD, netEMA = models.create_models(opt, model_config)
visualizer = tracking.visualizer(opt)
anatomy_sampler = OnlineAnatomySampler(
    max_displacement_frac=opt.anatomy_max_displacement
)
num_regions = model_config["num_mask_channels"]

data_iterator = iter(dataloader)
for index in range(opt.num_generated):
    batch = next(data_iterator)
    batch = utils.preprocess_real(
        batch, model_config["num_blocks_d0"], opt.device
    )
    sampled = anatomy_sampler.sample(
        batch["conditions"][:1], batch["masks"][:1], count=1
    )
    z_texture = torch.randn(
        1,
        num_regions,
        opt.texture_noise_dim,
        device=opt.device,
    )
    generator = netEMA if not opt.no_EMA else netG
    style_codes = generator.encode_style(
        batch["images"][-1][:1],
        batch["masks"][:1],
        output_count=1,
        randomize_patches=True,
    )
    fake = generator.generate(
        z_texture,
        conditions=sampled["conditions"],
        masks=sampled["masks"],
        style_codes=style_codes,
    )
    visualizer.save_batch(fake, opt.continue_epoch, i=str(index))
