from __future__ import annotations

from pathlib import Path
import importlib

import torch

from osada.model import BroadBandDomainAdaptNet
from osada.evaluation import parse_encoder_dilations


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIRED_MODULES = [
    "numpy",
    "torch",
    "sklearn",
    "matplotlib",
]
REQUIRED_FILES = [
    "data/source_gaussian_qpsk_rxshift_100mhz.pkl",
    "data/target_rayleigh_qpsk_rxshift_100mhz.pkl",
    "data/target_rician_qpsk_rxshift_100mhz.pkl",
    "weights/source_seed42.pth",
    "weights/osada_full_seed42.pth",
    "weights/ablation/osada_occ_seed42.pth",
    "weights/ablation/osada_occ_pseudo_seed42.pth",
    "weights/ablation/osada_full_seed42.pth",
]


def main() -> int:
    print("Checking Python modules...")
    for name in REQUIRED_MODULES:
        importlib.import_module(name)
        print(f"  OK: {name}")

    print("\nChecking project files...")
    for rel in REQUIRED_FILES:
        path = PROJECT_ROOT / rel
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"  OK: {rel}")

    print("\nChecking model forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BroadBandDomainAdaptNet(
        num_subbands=8,
        num_domains=2,
        fft_norm_mode="log_power",
        encoder_norm="bn",
        encoder_gn_groups=8,
        encoder_dilations=parse_encoder_dilations("1,1,1"),
        encoder_use_se=True,
        encoder_multiscale=False,
    ).to(device)
    model.eval()
    with torch.no_grad():
        logits, _domain, _aux = model(torch.randn(2, 2, 1024, device=device), grl_lambda=0.0, return_aux=True)
    if tuple(logits.shape) != (2, 8):
        raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)}")
    print(f"  OK: device={device}, output_shape={tuple(logits.shape)}")

    print("\nPC setup check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
