# OSMIS HierSPADE Quick Verification

This repository is configured for the current quick verification experiment on
one rendered pelvic-floor ultrasound C-plane image.  The goal is to test whether
the clean texture pathway from the earlier 31k online-minimal model can be kept
while hierarchical anatomy labels constrain the internal levator-hiatus
structure.

The repository trains on exactly one real image and one annotation.  No warped
image is treated as a real target, and no fixed 64-mask bank is used.

## Current default model

```text
z_global
    -> first_linear
    -> one OSMIS generator

0..5 hierarchical mask
    -> low/mid SPADE blocks only
    -> controls shape and internal anatomy layout

collapsed binary target/non-target mask
    + one real image style descriptor
    + z_texture
    + high-resolution noise injection
    -> high-resolution texture blocks
    -> controls appearance without giving class 3/4/5 appearance identities

output image + sampled 0..5 mask
```

This is intentionally not the previous full-SEAN hierarchy.  Full-SEAN allowed
small internal classes to become appearance regions and produced visible
semantic/color leakage.  In this quick model, the detailed 0..5 labels guide
only structure in the SPADE layers.  The texture path sees only target versus
non-target.

## Included data

Source assets:

```text
datasets/rendered_us_test2_source/
datasets/rendered_us_test2_multilevel_draft/
```

The indexed mask uses:

```text
0 background
1 rendered support outside target
2 levator-hiatus target remainder
3 anterior internal candidate
4 middle internal candidate
5 posterior internal candidate
```

Classes 3..5 are development annotations derived from intensity and position.
They are not independently validated clinical labels.

`run_5090.sh` prepares the training dataset automatically:

```text
datasets/rendered_us_test2_hierspade_quick/
```

This prepared directory is ignored by git because it is reproducible from the
tracked source image and indexed mask.

## Online anatomy sampling

Every batch receives a fresh mask condition.  The sampler applies bounded global
affine and smooth elastic deformation, then jitters internal candidates inside
the target.  It rejects masks with implausible area, centroid, or internal-area
changes.

The current defaults are deliberately less conservative than the earlier quick
test:

```text
max rotation:        +/- 7 degrees
scale_x:             0.84 .. 1.16
scale_y:             0.88 .. 1.12
translation:         +/- 0.09 normalized grid units
smooth displacement: 0.04 image fraction
accepted area ratio: 0.68 .. 1.42
centroid shift:      <= 0.10
internal jitter:     +/- 3 degrees, 0.88 .. 1.12 scale
support boundary:    weak transform only by default
```

These values are meant for verification, not final clinical validation.
For hierarchical masks, the rendered ultrasound support boundary is now sampled
with a separate weak transform so that the outer rendered volume remains mostly
stable while the levator-hiatus target and internal candidates keep larger
variation.

## RTX 5090 quick run

Create the environment once:

```bash
bash setup_5090.sh
conda activate osmis_multiprior_5090
```

Train the default quick experiment:

```bash
bash run_5090.sh
```

Defaults:

```text
experiment:   test2_hierspade_quick_5090
dataset:      rendered_us_test2_hierspade_quick
iterations:   5,000
batch size:   16
monitor/save: every 500 iterations
```

If a local 31k online-minimal checkpoint exists at:

```text
checkpoints/test2_online_minimal_31000_imported/models/
```

the script uses it for partial initialization.  If it is absent, training starts
from scratch and continues normally.

Override without editing:

```bash
NUM_EPOCHS=10000 BATCH_SIZE=8 EXP_NAME=my_hierspade_test bash run_5090.sh
```

Generate samples:

```bash
bash generate_5090.sh
```

Or choose a checkpoint:

```bash
bash generate_5090.sh 5000
```

## What to inspect

This quick experiment is useful only if all three are true:

- the image keeps the cleaner 31k-style rendered ultrasound texture;
- the internal structures begin to respond to the sampled 0..5 mask;
- class 3/4/5 do not leave color or annotation-like residue.

If the texture remains clean but the internal structure does not respond by
5k-10k iterations, low/mid SPADE alone is probably not enough and a stronger
structure objective or soft internal prior should be tested next.

## Provenance

- OSMIS: https://github.com/boschresearch/one-shot-synthesis
- SPADE reference: https://github.com/NVlabs/SPADE
- SEAN reference for earlier experiments: https://github.com/ZPdesu/SEAN

See [UPSTREAM.md](UPSTREAM.md), [LICENSE](LICENSE), and
[3rd-party-licenses.txt](3rd-party-licenses.txt).

## Pelvic test6/test7 sequential training

The repository includes source images for two additional cases:

```text
datasets/pelvic_test6_source/image/00000.jpg
datasets/pelvic_test7_source/image/00000.jpg
```

Before training, add one binary target mask for each case:

```text
datasets/pelvic_test6_source/mask/00000.png
datasets/pelvic_test7_source/mask/00000.png
```

The binary mask should be black/white, with white = levator-hiatus segmentation
target.  The script derives the rendered support and 0..5 multi-class anatomy
condition automatically.

Train test6 and test7 sequentially:

```bash
bash run_pelvic_test6_test7_5090.sh
```

Useful overrides:

```bash
NUM_EPOCHS=150000 SAVE_FREQ=1000 bash run_pelvic_test6_test7_5090.sh
CASES="test6" NUM_EPOCHS=100000 bash run_pelvic_test6_test7_5090.sh
CONTINUE=1 NUM_EPOCHS=200000 bash run_pelvic_test6_test7_5090.sh
```

Generate after training:

```bash
bash generate_pelvic_case_5090.sh test6 100000
bash generate_pelvic_case_5090.sh test7 100000
```
