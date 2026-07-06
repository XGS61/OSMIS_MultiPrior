# OSMIS Full-SEAN Hierarchical One-Shot Ultrasound

This repository trains on exactly one rendered pelvic-floor ultrasound C-plane
image and one review annotation. OSMIS remains the single generator and
discriminator backbone. There is no SinGAN pyramid, offline pseudo-image set,
or fixed mask bank.

## Current model

```text
one hierarchical annotation
        + random shape parameters
        -> online topology-aware condition sampler
        -> fresh valid structural condition M(u)

one real image + exclusive semantic regions
        -> regional real-patch style encoder
        -> per-region style codes

learned OSMIS input
        -> one OSMIS generator
        -> every OSMIS G-block uses full SEAN-style dual modulation
           (hierarchical mask branch + regional style branch)
        -> generated image

regional z_texture changes texture only
M(u) determines anatomy and is saved as the output annotation
```

The discriminator is the single OSMIS discriminator with low-level, masked
content, and weak layout branches. It is used only during training.

## What is different from the previous checkpoint

- All OSMIS generator blocks now use full SEAN-style dual modulation.
- The former unconstrained `z_global` is replaced by a learned input tensor.
- Anatomy variation comes only from the validated hierarchical condition.
- `z_texture` is sampled independently for all six exclusive style regions.
- The fixed global-statistics style code is replaced by real regional patch
  style sampling from the one source image.
- The ineffective discriminator latent regressor is removed.
- A single region-restricted, high-frequency mode-seeking objective makes
  `z_texture` observable without rewarding low-frequency anatomy changes.

Old 31k and 99k weights are architecturally incompatible and cannot be resumed.

## Included annotation draft

The source assets are:

```text
datasets/rendered_us_test2_source/
datasets/rendered_us_test2_multilevel_draft/
```

The hierarchy contains seven structural channels:

```text
0 rendered support
1 supplied levator-hiatus annotation
2 anterior internal candidate
3 middle internal candidate
4 posterior internal candidate
5 hiatus boundary
6 signed hiatus distance
```

The indexed visualization has six exclusive style regions:

```text
0 outside support
1 rendered tissue
2 hiatus remainder
3 anterior candidate
4 middle candidate
5 posterior candidate
```

The internal candidates were derived from position and intensity for model
development. They are review annotations, not independently validated clinical
ground truth. Do not report them as clinical labels without expert review.

`tools/create_multilevel_mask_draft.py` reproduces the draft. The training
pipeline uses `hierarchical_conditions.npz`, which preserves parent-child
overlap instead of flattening the hierarchy.

## Online mask generation

Every training batch and inference sample receives a fresh condition:

1. apply one bounded global affine and smooth elastic field to support, hiatus,
   and internal structures;
2. apply smaller relative transformations to internal candidates;
3. enforce containment and non-overlap;
4. preserve anterior-middle-posterior ordering;
5. reject invalid area or centroid changes;
6. regenerate exclusive regions, boundary, and signed-distance channels.

This process creates conditions only. It never warps the source image or adds
any pseudo-image to the real training distribution.

## Generator conditioning

All stages belong to one OSMIS generator:

```text
5 -> 10 -> 20 -> 40 -> 80 -> 160 -> 320 pixels
```

This is not a set of independently trained scale GANs. The same condition is
resized continuously for each OSMIS block:

- low resolution: mask branch dominates global anatomy;
- middle resolution: mask and style are balanced;
- high resolution: regional style is stronger while mask control remains.

## Objective

```text
L_G = L_OSMIS-adversarial
    + lambda_structure * L_structure
    + lambda_diversity * L_high-frequency-region-diversity
    + lambda_anchor(t) * L_anchor
```

The diversity loss compares two images generated from the same condition and
the same reference style but different regional texture latents. It operates
on high-pass image content inside each semantic region. The anchor and layout
weights decay after warm-up.

## RTX 5090

Create the environment once:

```bash
bash setup_5090.sh
conda activate osmis_multiprior_5090
```

Train the included case:

```bash
bash run_5090.sh
```

Defaults:

- experiment: `test2_fullsean_hierarchical_5090`
- image size: 320 x 320
- batch size: 16
- iterations: 100,000
- checkpoint and monitor interval: 1,000

Override without editing:

```bash
NUM_EPOCHS=5000 BATCH_SIZE=8 EXP_NAME=fullsean_smoke bash run_5090.sh
```

Generate 50 samples from the latest checkpoint:

```bash
bash generate_5090.sh
```

Or choose a checkpoint:

```bash
bash generate_5090.sh 50000
```

Each generated image is accompanied by the six-class label map and a
`*_primary_mask.png` binary downstream target. The primary target is the union
of label values `2..5`, so adding internal conditions does not change the
original levator-hiatus segmentation task.

## Monitoring

Every monitor contains:

- first eight images: identical anatomy and reference style, varied
  region-specific `z_texture`;
- last eight images: fixed `z_texture`, freshly sampled valid anatomy.

This separates texture collapse from condition collapse.

## Scope and limitation

The implementation can enforce explicit topology and produce controlled
variation around one annotated case. A single case cannot identify the true
population distribution or validate the anatomical identity of automatically
proposed internal regions. Clinical validity and downstream segmentation gain
must be evaluated independently.

## Provenance

- OSMIS: https://github.com/boschresearch/one-shot-synthesis
- SEAN reference: https://github.com/ZPdesu/SEAN
- SPADE reference: https://github.com/NVlabs/SPADE

See [UPSTREAM.md](UPSTREAM.md), [LICENSE](LICENSE), and
[3rd-party-licenses.txt](3rd-party-licenses.txt).
