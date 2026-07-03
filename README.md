# OSMIS-MultiPrior

One-shot generation for a rendered 3D pelvic-floor ultrasound C-plane image.
The repository trains on **exactly one real image/mask pair** and uses
anatomy-bounded masks only as generator conditions.

This is deliberately different from rebuilding a dataset of warped images:

- `image/` contains exactly one real image.
- `mask/` contains its one manual levator-hiatus mask.
- `mask_priors/` contains varied conditions only.
- no deformed image is presented to the discriminator as real;
- only one fixed latent and the original mask use an anchor reconstruction loss.

## Model

The generator retains the OSMIS multi-scale backbone:

1. low-resolution blocks learn global ultrasound layout from latent noise;
2. middle blocks use SPADE with an anatomy-bounded levator-hiatus mask;
3. high-resolution blocks use SEAN-inspired regional style statistics extracted
   from the single real image, plus latent style perturbation and noise injection.

The discriminator sees RGB images, not a concatenated full mask. The mask is
used only for OSMIS regional content attention. Training adds regional texture
distribution, frequency distribution, mask-boundary contrast/gradient,
same-mask diversity, and a single reconstruction-anchor loss.

The implementation is based on the open-source OSMIS code and the lightweight
SPADE version previously validated in `OSMIS_Improved`. See [UPSTREAM.md](UPSTREAM.md)
and [3rd-party-licenses.txt](3rd-party-licenses.txt).

## Included data

The latest requested pair is included:

- `datasets/rendered_us_test2_source/image/00000.jpg`
- `datasets/rendered_us_test2_source/mask/00000.png`

The preparation script crops the top 20 acquisition-marker pixels from both
files, validates alignment, saves one real pair, and builds 64 mask-only priors.

## RTX 5090: one-command training

First create the environment once:

```bash
bash setup_5090.sh
conda activate osmis_multiprior_5090
```

Then train:

```bash
bash run_5090.sh
```

Defaults are batch size 16, 100,000 iterations, and a checkpoint/monitor image
every 1,000 iterations. Override without editing files, for example:

```bash
NUM_EPOCHS=5000 BATCH_SIZE=8 EXP_NAME=smoke_test2 bash run_5090.sh
```

Despite the inherited option name, `num_epochs` means optimizer iterations.
Training output is written to `run_logs/<EXP_NAME>/train.log`; weights and
monitor grids are under `checkpoints/<EXP_NAME>/`.

Each monitor grid contains:

- first 8 images: the same mask with different latent codes;
- last 8 images: different masks with the same latent code.

## Generate

The latest checkpoint is selected automatically:

```bash
bash generate_5090.sh
```

Or choose an iteration:

```bash
bash generate_5090.sh 50000
```

Images and their binary condition masks are saved to
`checkpoints/test2_multiprior_5090/evaluation/<ITERATION>/`.

## Use another single case

Supply aligned image and binary mask paths:

```bash
IMAGE_PATH=/path/case.jpg MASK_PATH=/path/case_mask.png \
DATASET_NAME=my_case EXP_NAME=my_case_run bash run_5090.sh
```

Change `crop-top` in `run_5090.sh` if the new image has no top marker. A mask is
required for the current guided model; the script never manufactures real
training images from it.

## Important limitation

Mask priors express bounded geometric hypotheses around one annotation; they
are not learned population anatomy. Outputs still require quantitative and
expert review before use in a segmentation study.
