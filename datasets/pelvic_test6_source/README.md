Single-case source for pelvic test6.

Expected files:

- `image/00000.jpg`: source rendered pelvic-floor ultrasound image.
- `mask/00000.png`: user-provided binary target mask, white = levator-hiatus / segmentation target, black = everything else.

After `mask/00000.png` is added, run `bash run_pelvic_test6_test7_5090.sh`.
The script derives the rendered support and multi-class 0..5 anatomy condition automatically.
