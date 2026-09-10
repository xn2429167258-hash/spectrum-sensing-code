from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evaluate_main import (
    DEFAULT_RAYLEIGH_DATA,
    DEFAULT_SNR_LIST,
    evaluate,
    make_loader,
    make_model,
    parse_snr_list,
    resolve_path,
)
from osada.data_utils import set_random_seed


DEFAULT_CKPTS = {
    "DA-Occ": "weights/ablation/osada_occ_seed42.pth",
    "Occ+Pseudo": "weights/ablation/osada_occ_pseudo_seed42.pth",
    "Full OSADA": "weights/ablation/osada_full_seed42.pth",
}


def compact_row(name: str, domain_name: str, checkpoint: Path, metrics: dict):
    return {
        "output": name,
        "eval_domain": domain_name,
        "f1_macro": f"{metrics['f1_macro']:.6f}",
        "subset_acc": f"{metrics['subset_accuracy_exact_match']:.6f}",
        "checkpoint": str(checkpoint.relative_to(Path(__file__).resolve().parent)),
    }



def print_per_snr(per_snr_rows):
    print("\nPer-SNR results:")
    print(f"{'Output':12s} {'SNR':>4s} {'F1':>9s} {'Subset':>9s}")
    for row in per_snr_rows:
        print(f"{row['output']:12s} {str(row['snr']):>4s} {float(row['f1_macro']):9.6f} {float(row['subset_acc']):9.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test three pretrained OSADA ablation checkpoints. No training is performed.")
    parser.add_argument("--target-path", default=DEFAULT_RAYLEIGH_DATA)
    parser.add_argument("--domain-name", default="Rayleigh")
    parser.add_argument("--modulation", default="QPSK")
    parser.add_argument("--snr-list", default=",".join(str(v) for v in DEFAULT_SNR_LIST))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-subbands", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--subband-reweight-mode", choices=["none", "soft"], default="soft")
    parser.add_argument("--fft-norm-mode", choices=["log_power", "zscore", "center"], default="log_power")
    parser.add_argument("--encoder-norm", choices=["bn", "gn", "in", "none"], default="bn")
    parser.add_argument("--encoder-gn-groups", type=int, default=8)
    parser.add_argument("--encoder-dilations", default="1,1,1")
    parser.add_argument("--no-encoder-use-se", action="store_true")
    parser.add_argument("--encoder-multiscale", action="store_true")
    parser.add_argument("--no-input-unit-energy", action="store_true")
    parser.add_argument("--no-print-per-snr", action="store_true", help="Do not print the per-SNR table to stdout.")
    parser.add_argument("--seed", type=int, default=42)
    for name, default in DEFAULT_CKPTS.items():
        dest = name.lower().replace("+", "_").replace("-", "_").replace(" ", "_") + "_checkpoint"
        cli = name.lower().replace("+", "-").replace(" ", "-") + "-checkpoint"
        parser.add_argument(f"--{cli}", dest=dest, default=default)
    args = parser.parse_args()

    args.target_path = resolve_path(args.target_path)
    args.snr_values = parse_snr_list(args.snr_list)
    args.checkpoints = {}
    for name in DEFAULT_CKPTS:
        opt = name.lower().replace("+", "_").replace("-", "_").replace(" ", "_") + "_checkpoint"
        args.checkpoints[name] = resolve_path(getattr(args, opt))
    return args


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Testing pretrained ablation checkpoints only. No training and no data generation.")

    target_loader, target_snr = make_loader(args.target_path, args)

    rows = []
    per_snr_rows = []
    for name in ["DA-Occ", "Occ+Pseudo", "Full OSADA"]:
        checkpoint = args.checkpoints[name]
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = make_model(checkpoint, args, device)
        metrics, per_snr = evaluate(model, target_loader, target_snr, args, device)
        row = compact_row(name, args.domain_name, checkpoint, metrics)
        rows.append(row)
        for item in per_snr:
            per_snr_rows.append({
                "output": name,
                "eval_domain": args.domain_name,
                "snr": item["snr"],
                "f1_macro": f"{item['f1_macro']:.6f}",
                "subset_acc": f"{item['subset_acc']:.6f}",
            })
        print(f"{name:12s} F1={row['f1_macro']} Subset={row['subset_acc']}", flush=True)

    if not args.no_print_per_snr:
        print_per_snr(per_snr_rows)



if __name__ == "__main__":
    main()


