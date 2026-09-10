from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from osada.model import BroadBandDomainAdaptNet  # noqa: E402
from osada.evaluation import (  # noqa: E402
    compute_overall_metrics,
    compute_per_snr_metrics,
    parse_encoder_dilations,
    predict_with_thresholds,
    run_inference,
)
from osada.data_utils import (  # noqa: E402
    build_dataloader,
    load_source_data,
    set_random_seed,
    split_source_train_val_test,
)


DEFAULT_SNR_LIST = [-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
DEFAULT_SOURCE_CKPT = "weights/source_seed42.pth"
DEFAULT_OSADA_CKPT = "weights/osada_full_seed42.pth"
DEFAULT_SOURCE_DATA = "data/source_gaussian_qpsk_rxshift_100mhz.pkl"
DEFAULT_RAYLEIGH_DATA = "data/target_rayleigh_qpsk_rxshift_100mhz.pkl"
DEFAULT_RICIAN_DATA = "data/target_rician_qpsk_rxshift_100mhz.pkl"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_snr_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def make_loader(data_path: Path, args):
    x_data, y_data, snrs = load_source_data(
        str(data_path),
        modulation=args.modulation,
        snr_list=args.snr_values,
        normalize_iq=not args.no_input_unit_energy,
    )
    _x_train, _x_val, x_test, _y_train, _y_val, y_test, _snr_train, _snr_val, snr_test = split_source_train_val_test(
        x_data,
        y_data,
        snrs,
        seed=args.seed,
    )
    return build_dataloader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False), snr_test


def make_model(checkpoint: Path, args, device: torch.device):
    model = BroadBandDomainAdaptNet(
        num_subbands=args.num_subbands,
        num_domains=2,
        fft_norm_mode=args.fft_norm_mode,
        encoder_norm=args.encoder_norm,
        encoder_gn_groups=args.encoder_gn_groups,
        encoder_dilations=parse_encoder_dilations(args.encoder_dilations),
        encoder_use_se=not args.no_encoder_use_se,
        encoder_multiscale=args.encoder_multiscale,
    ).to(device)
    state_dict = torch.load(str(checkpoint), map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            f"[warn] {checkpoint.name}: missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    model.eval()
    return model


def evaluate(model, loader, snr_test, args, device):
    probs, labels = run_inference(
        model,
        loader,
        device,
        subband_reweight_mode=args.subband_reweight_mode,
    )
    thresholds = np.full(args.num_subbands, args.threshold, dtype=np.float32)
    preds = predict_with_thresholds(probs, thresholds)
    return compute_overall_metrics(labels, preds), compute_per_snr_metrics(labels, preds, snr_test)


def row_from_metrics(name: str, domain: str, checkpoint: Path, metrics: dict):
    return {
        "output": name,
        "eval_domain": domain,
        "f1_macro": f"{metrics['f1_macro']:.6f}",
        "pd_macro": f"{metrics['pd_macro_recall']:.6f}",
        "precision_macro": f"{metrics['precision_macro']:.6f}",
        "subset_acc": f"{metrics['subset_accuracy_exact_match']:.6f}",
        "element_acc": f"{metrics['subband_accuracy_element_level']:.6f}",
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
    }


def print_overall(rows):
    print("\nOverall results:")
    print(f"{'Output':14s} {'Domain':8s} {'F1':>9s} {'Pd':>9s} {'Subset':>9s}")
    for row in rows:
        print(
            f"{row['output']:14s} {row['eval_domain']:8s} "
            f"{float(row['f1_macro']):9.6f} {float(row['pd_macro']):9.6f} "
            f"{float(row['subset_acc']):9.6f}"
        )


def print_per_snr(per_snr_rows):
    print("\nPer-SNR results:")
    print(f"{'Output':14s} {'Domain':8s} {'SNR':>4s} {'Pd':>9s} {'F1':>9s} {'Subset':>9s}")
    for row in per_snr_rows:
        print(
            f"{row['output']:14s} {row['eval_domain']:8s} {str(row['snr']):>4s} "
            f"{float(row['pd_macro']):9.6f} {float(row['f1_macro']):9.6f} "
            f"{float(row['subset_acc']):9.6f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Test the three pretrained outputs used by our OSADA paper package.")
    parser.add_argument("--source-checkpoint", default=DEFAULT_SOURCE_CKPT)
    parser.add_argument("--osada-checkpoint", default=DEFAULT_OSADA_CKPT)
    parser.add_argument("--source-path", default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--rayleigh-path", default=DEFAULT_RAYLEIGH_DATA)
    parser.add_argument("--rician-path", default=DEFAULT_RICIAN_DATA)
    parser.add_argument("--eval-domains", choices=["all", "rayleigh", "rician"], default="all")
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
    args = parser.parse_args()

    args.source_checkpoint = resolve_path(args.source_checkpoint)
    args.osada_checkpoint = resolve_path(args.osada_checkpoint)
    args.source_path = resolve_path(args.source_path)
    args.rayleigh_path = resolve_path(args.rayleigh_path)
    args.rician_path = resolve_path(args.rician_path)
    args.snr_values = parse_snr_list(args.snr_list)
    return args


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Testing pretrained outputs only. No training and no data generation.")

    specs = []
    if args.eval_domains in {"all", "rayleigh"}:
        rayleigh_loader, rayleigh_snr = make_loader(args.rayleigh_path, args)
        specs.extend([
            ("Source-only", "Rayleigh", args.source_checkpoint, rayleigh_loader, rayleigh_snr),
            ("OSADA-Rayleigh", "Rayleigh", args.osada_checkpoint, rayleigh_loader, rayleigh_snr),
        ])
    if args.eval_domains in {"all", "rician"}:
        rician_loader, rician_snr = make_loader(args.rician_path, args)
        specs.append(("OSADA-Rician", "Rician", args.osada_checkpoint, rician_loader, rician_snr))

    rows = []
    per_snr_rows = []
    for name, domain, checkpoint, loader, snr_test in specs:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = make_model(checkpoint, args, device)
        metrics, per_snr = evaluate(model, loader, snr_test, args, device)
        row = row_from_metrics(name, domain, checkpoint, metrics)
        rows.append(row)
        for item in per_snr:
            item = dict(item)
            item["output"] = name
            item["eval_domain"] = domain
            per_snr_rows.append(item)
    print_overall(rows)

    if not args.no_print_per_snr:
        print_per_snr(per_snr_rows)



if __name__ == "__main__":
    main()
