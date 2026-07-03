"""Fail fast if the active PyTorch build cannot execute on an RTX 5090."""

import sys

import torch


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is unavailable. Run bash setup_5090.sh first.", file=sys.stderr)
        return 1
    capability = torch.cuda.get_device_capability()
    cuda_version = tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2])
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}; capability: sm_{capability[0]}{capability[1]}")
    if capability >= (12, 0) and cuda_version < (12, 8):
        print("ERROR: RTX 5090 needs a CUDA 12.8+ PyTorch build.", file=sys.stderr)
        return 2
    probe = torch.randn(512, 512, device="cuda")
    value = (probe @ probe.T).mean()
    torch.cuda.synchronize()
    print(f"CUDA matrix test: OK ({value.item():.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
