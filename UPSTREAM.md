# Upstream provenance

- Repository: https://github.com/boschresearch/one-shot-synthesis
- Imported commit: `1f980de1909e27848fa48c54f16cd8d8e2fd3fac`
- Paper: *One-Shot Synthesis of Images and Segmentation Masks*, WACV 2023
- License: AGPL-3.0

The upstream `LICENSE`, `3rd-party-licenses.txt`, and `environment.yml` are
preserved in this directory. The README documents this derived implementation.

This derived repository keeps one OSMIS generator and discriminator while
adding topology-aware hierarchical online conditions, full SEAN-style
mask/style modulation in every OSMIS generator block, regional real-patch style
encoding, and region-specific texture latents. The SEAN design was implemented
for this repository with the CVPR 2020 paper and the public reference at
https://github.com/ZPdesu/SEAN as architectural references. It intentionally
does not train on warped pseudo-images or a fixed offline mask bank.
