# OSMIS Online Anatomy

One-shot generation for a rendered pelvic-floor ultrasound C-plane image.
The model uses exactly one real image/annotation pair. It does not construct
warped pseudo-images and does not use a fixed offline mask bank.

## What changed

- OSMIS remains the only image-generation backbone.
- Middle generator blocks use SPADE for anatomy conditions.
- High-resolution blocks use SEAN-inspired regional style modulation.
- A fresh anatomy condition is sampled online for every training batch and
  every generated image.
- Global/layout and regional texture latents are independent.
- A latent-regression head on shared discriminator features forces the
  generator to use the texture latent.
- The training objective is intentionally limited to four families:
  adversarial, structure consistency, latent reconstruction, and a decaying
  single reconstruction anchor.

The online sampler is non-neural. It applies bounded affine and smooth elastic
changes to the annotation, validates area and centroid displacement, and never
creates a corresponding fake "real" image.

## Annotation requirements

The included example currently has a binary annotation:

```text
0   background
255 annotated outer region
```

This mode runs directly, but it can only control that one annotated region.
It cannot guarantee independent, clinically correct variation of an unlabelled
internal hiatus.

For independent internal-structure control, supply one indexed PNG label map:

```text
0 background
1 outer/primary annotated region
2 internal hiatus
3..K optional additional internal structures
```

Pixel values must be literal class indices `0, 1, 2, ...`, not display colors.
The additional labels are generation-only conditions; the downstream
segmentation target may still be reduced to the desired primary class.

When separate binary outer/inner masks are available, merge them with:

```bash
python build_multilevel_annotation.py \
  --outer current_mask.png \
  --inner internal_hiatus.png \
  --output indexed_anatomy.png
```

All classes receive the same smooth global transform. Internal classes receive
a smaller relative transform and are constrained to remain inside the global
foreground support.

## Model

At 320 x 320, the generator follows:

```text
z_global -> low-resolution OSMIS blocks
online anatomy condition -> middle SPADE blocks
single real image + z_texture -> high-resolution SEAN-style blocks
```

The discriminator retains OSMIS low-level, masked-content, and weak layout
branches. A small latent head regresses `z_texture` from shared downsampled
features.

The generator objective is:

```text
L_G = L_adversarial
    + lambda_structure * L_structure
    + lambda_latent * L_latent
    + lambda_anchor(t) * L_anchor
```

The layout and anchor weights decay after warm-up. There are no separate
texture-statistics, frequency, same-mask pixel-diversity, or offline-mask losses.

## RTX 5090

Create the environment once:

```bash
bash setup_5090.sh
conda activate osmis_multiprior_5090
```

Train:

```bash
bash run_5090.sh
```

Defaults:

- source pair: `datasets/rendered_us_test2_source/`
- experiment: `test2_online_minimal_5090`
- batch size: 16
- iterations: 100,000
- checkpoints and monitor grids: every 1,000 iterations

Override without editing:

```bash
NUM_EPOCHS=5000 BATCH_SIZE=8 EXP_NAME=online_smoke bash run_5090.sh
```

Training creates a new output name and does not overwrite previous 99k
MultiPrior checkpoints.

## Generate

Use the latest checkpoint:

```bash
bash generate_5090.sh
```

Or specify an iteration:

```bash
bash generate_5090.sh 50000
```

Every result receives a newly sampled online condition and fresh global/texture
latents. The condition mask saved beside the image is the actual condition used
for that sample.

## Monitor layout

Each training monitor contains 16 images:

- first 8: identical anatomy and global latent, different texture latents;
- last 8: identical latents, eight freshly sampled anatomy conditions.

This makes latent collapse and condition collapse visible during training.

## Limitation

One case cannot identify a population distribution of unlabelled anatomy.
Online sampling provides continuous variation within explicit constraints; it
does not manufacture unknown clinical knowledge. Internal anatomy should not be
claimed as controlled until the corresponding auxiliary label or landmark has
been supplied and independently evaluated.

## Provenance

The code derives from the open-source OSMIS implementation:
https://github.com/boschresearch/one-shot-synthesis

See [UPSTREAM.md](UPSTREAM.md), [LICENSE](LICENSE), and
[3rd-party-licenses.txt](3rd-party-licenses.txt).
