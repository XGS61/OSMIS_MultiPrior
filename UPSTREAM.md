# Upstream provenance

- Repository: https://github.com/boschresearch/one-shot-synthesis
- Imported commit: `1f980de1909e27848fa48c54f16cd8d8e2fd3fac`
- Paper: *One-Shot Synthesis of Images and Segmentation Masks*, WACV 2023
- License: AGPL-3.0

The upstream `LICENSE`, `3rd-party-licenses.txt`, and `environment.yml` are
preserved in this directory. The README documents this derived implementation.

This derived repository adds mask-only anatomy priors, selective SPADE,
SEAN-inspired regional texture modulation, RGB-only low-level discrimination,
and distributional texture/anatomy losses. It intentionally does not train on
warped pseudo-images. Details are documented in `README.md`.
