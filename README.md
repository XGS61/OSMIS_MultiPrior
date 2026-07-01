# OSMIS Improved: anatomy-guided one-shot pelvic-floor ultrasound synthesis

This repository preserves the OSMIS backbone while changing the generation
direction from unconditional joint image-mask synthesis to target-mask
conditioned image synthesis.

## What changed

- The original OSMIS code remains untouched in the sibling `OSMIS` directory.
- A conservative C-plane pseudo-pair generator creates smooth image/mask warps.
- Candidate masks must remain a single, hole-free, pear-shaped levator-hiatus
  region with bounded area, dimensions, centroid, and SP/PVM endpoint shifts.
- SPADE blocks inject the target mask at every generator scale.
- The discriminator receives the target mask together with each image scale.
- The former generated mask is now an auxiliary prediction used for Dice and
  boundary consistency. The supplied target mask is the output annotation.
- Anatomy-breaking layout augmentation (object move/copy/delete and large
  translations/crops) is disabled.
- Low-frequency anatomy and region-wise texture statistics losses are added.

This is a minimum effective version. It does not claim to model population
anatomy or transitions between rest, contraction, and Valsalva from one case.

## Segmentation target

The binary mask represents the **interior region of the levator hiatus** on the
oblique axial C-plane at the level of minimal anteroposterior hiatal dimensions.
The superior limit corresponds to the posterior aspect of the symphysis pubis
(SP), and the inferior limit to the anterior border of the pubovisceral muscle
(PVM). In this first version these landmarks are inferred from the superior and
inferior mask endpoints and used as conservative deformation anchors. They can
also be supplied manually to `prepare_anatomy_dataset.py` as `--sp x,y` and
`--pvm x,y`.

The implementation follows the constraints described in:

- Sindhwani et al., *Semi-automatic outlining of levator hiatus*, UOG 2016.
- Bonmati et al., *Automatic segmentation method of pelvic floor levator
  hiatus in ultrasound using a self-normalizing neural network*, JMI 2018.
- SPADE conditioning: https://github.com/NVlabs/SPADE
- OSMIS upstream: https://github.com/boschresearch/one-shot-synthesis

The two pelvic-floor papers do not provide a public implementation. The smooth
deformation code therefore uses a transparent SciPy cubic control-grid field,
following the same conservative B-spline/TPS family of ideas rather than
claiming to reproduce their BEAS optimizer.

## Environment

```bash
conda create -n osmis_improved python=3.10 -y
conda activate osmis_improved
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-modern.txt
```

For a different CUDA/PyTorch combination, install the matching official
PyTorch wheel before the remaining requirements.

## One-command training

The repository includes the current example pair under
`datasets/rendered_us_3d_1/`.

```bash
bash train_improved.sh
```

The default run uses 32 validated pseudo-pairs, batch size 8, 150,000
iterations, and saves checkpoints and previews every 1,000 iterations.

Override settings without editing the script:

```bash
NUM_EPOCHS=10000 BATCH_SIZE=8 NUM_VARIANTS=32 bash train_improved.sh my_test
```

Use another image/mask pair:

```bash
IMAGE_PATH=/path/to/image.png MASK_PATH=/path/to/mask.png bash train_improved.sh my_case
```

## Generate

```bash
bash generate_improved.sh rendered_us_atg_osmis_v1 150000 50
```

Outputs are written to:

```text
checkpoints/<experiment>/evaluation/<epoch>/
```

Each sample includes the generated image, supplied target mask, auxiliary
predicted mask, and raw label maps.

## Quick implementation check

```bash
bash run_smoke_test.sh
```

Two iterations only verify the data and model paths; they are not a meaningful
training result.
