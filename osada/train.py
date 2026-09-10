import argparse
import csv
import datetime
import os
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from osada.model import BroadBandDomainAdaptNet
from osada.source_training import (
    train_phase1_source_cls,
    train_phase2_cls,
    train_phase2_finetune,
    train_phase3_da,
)
from osada.data_utils import (
    DEVICE,
    build_dataloader,
    load_source_data,
    set_random_seed,
    split_source_train_val_test,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")


def _data_path(filename):
    return os.path.join(DATA_DIR, filename)


_SOURCE_GAUSSIAN = _data_path("source_gaussian_qpsk_rxshift_100mhz.pkl")
_TARGET_RAYLEIGH = _data_path("target_rayleigh_qpsk_rxshift_100mhz.pkl")
_TARGET_RICIAN = _data_path("target_rician_qpsk_rxshift_100mhz.pkl")


CONFIG = {
    "GAUSSIAN_DATA_PATH": _SOURCE_GAUSSIAN,
    "GAUSSIAN_SOURCE_PATH": _SOURCE_GAUSSIAN,
    "RICIAN_SOURCES": {
        "mild": _TARGET_RICIAN,
        "medium": _TARGET_RICIAN,
        "hard": _TARGET_RICIAN,
    },
    "RAYLEIGH_SOURCES": {
        "mild": _TARGET_RAYLEIGH,
        "medium": _TARGET_RAYLEIGH,
        "hard": _TARGET_RAYLEIGH,
    },
    "RICIAN_TARGETS": {
        "mild": _TARGET_RICIAN,
        "medium": _TARGET_RICIAN,
        "hard": _TARGET_RICIAN,
    },
    "RAYLEIGH_TARGETS": {
        "mild": _TARGET_RAYLEIGH,
        "medium": _TARGET_RAYLEIGH,
        "hard": _TARGET_RAYLEIGH,
    },
    "MODULATION": "QPSK",
    "BATCH_SIZE": 32,
    "SEED": 42,
    "EPOCHS_PHASE1": 30,
    "EPOCHS_PHASE2": 40,
    "EPOCHS_PHASE2_FINETUNE": 10,
    "EPOCHS_PHASE3": 30,
    "LR_PHASE1": 1e-3,
    "LR_PHASE2": 1e-3,
    "LR_FINETUNE_ENCODER": 1e-5,
    "LR_FINETUNE_CLASSIFIER": 1e-4,
    "LR_PHASE3": 1e-4,
    "FINETUNE_UNFREEZE_LAST_BLOCKS": 1,
    "TRAIN_SNR_LIST": [-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    "SAVE_ROOT": RUNS_DIR,
    "MODEL_TAG": "Model3FreqOnlyNPCFAR",
    "FFT_NORM_MODE": "log_power",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Model3: occupancy-semantic adversarial domain adaptation"
    )
    parser.add_argument("--mode", type=str, required=True, choices=["basic", "da"])
    parser.add_argument(
        "--source-family",
        type=str,
        default="gaussian",
        choices=["gaussian", "rician", "rayleigh"],
    )
    parser.add_argument(
        "--source-domain",
        type=str,
        default="hard",
        choices=["mild", "medium", "hard"],
        help="Used when source-family is rician or rayleigh.",
    )
    parser.add_argument("--source-path", type=str, default=None)
    parser.add_argument(
        "--target-family",
        type=str,
        default="rician",
        choices=["rician", "rayleigh"],
    )
    parser.add_argument("--target-domain", type=str, default="hard", choices=["mild", "medium", "hard"])
    parser.add_argument("--target-path", type=str, default=None)
    parser.add_argument("--base-checkpoint", type=str, default=None)
    parser.add_argument("--save-root", type=str, default=CONFIG["SAVE_ROOT"])
    parser.add_argument("--seed", type=int, default=CONFIG["SEED"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["BATCH_SIZE"])
    parser.add_argument("--epochs-phase1", type=int, default=CONFIG["EPOCHS_PHASE1"])
    parser.add_argument("--epochs-phase2", type=int, default=CONFIG["EPOCHS_PHASE2"])
    parser.add_argument("--epochs-phase2-finetune", type=int, default=CONFIG["EPOCHS_PHASE2_FINETUNE"])
    parser.add_argument("--epochs-phase3", type=int, default=CONFIG["EPOCHS_PHASE3"])
    parser.add_argument("--lr-phase1", type=float, default=CONFIG["LR_PHASE1"])
    parser.add_argument("--lr-phase2", type=float, default=CONFIG["LR_PHASE2"])
    parser.add_argument("--lr-finetune-encoder", type=float, default=CONFIG["LR_FINETUNE_ENCODER"])
    parser.add_argument("--lr-finetune-classifier", type=float, default=CONFIG["LR_FINETUNE_CLASSIFIER"])
    parser.add_argument("--lr-phase3", type=float, default=CONFIG["LR_PHASE3"])
    parser.add_argument("--finetune-unfreeze-last-blocks", type=int, default=CONFIG["FINETUNE_UNFREEZE_LAST_BLOCKS"])
    parser.add_argument("--model-tag", type=str, default="Model3_OSADA")
    parser.add_argument(
        "--fft-norm-mode",
        type=str,
        default=CONFIG["FFT_NORM_MODE"],
        choices=["log_power", "zscore", "center"],
        help="FFT preprocessing inside Model3. log_power preserves more amplitude information than per-sample zscore.",
    )
    parser.add_argument(
        "--encoder-norm",
        type=str,
        default="bn",
        choices=["bn", "gn", "in", "none"],
        help="Normalization inside the subband frequency encoder.",
    )
    parser.add_argument(
        "--encoder-gn-groups",
        type=int,
        default=8,
        help="Number of groups used by GroupNorm when --encoder-norm gn.",
    )
    parser.add_argument(
        "--encoder-dilations",
        type=str,
        default="1,1,1",
        help="Comma-separated dilation values for the three residual blocks, e.g. 1,2,4.",
    )
    parser.add_argument(
        "--encoder-use-se",
        action="store_true",
        help="Enable lightweight squeeze-and-excitation in each residual block.",
    )
    parser.add_argument(
        "--encoder-multiscale",
        action="store_true",
        help="Use a lightweight multi-kernel first convolution inside each residual block.",
    )
    parser.add_argument(
        "--no-input-unit-energy",
        action="store_true",
        help="Disable per-sample IQ unit-energy normalization when loading data.",
    )
    parser.add_argument(
        "--da-no-detach-condition",
        action="store_true",
    )
    parser.add_argument(
        "--da-no-occupancy-condition",
        action="store_true",
        help="Disable occupancy-conditioned domain input.",
    )
    parser.add_argument(
        "--da-subband-reweight-mode",
        type=str,
        default="soft",
        choices=["none", "soft"],
        help="Subband-aware feature reweighting mode for the frequency branch.",
    )
    parser.add_argument(
        "--source-subband-reweight-mode",
        type=str,
        default="soft",
        choices=["none", "soft"],
        help="Subband-aware reweighting mode used during source-only/base training.",
    )
    parser.add_argument("--source-aux-weight", type=float, default=0.3)
    parser.add_argument(
        "--source-pos-weight",
        type=float,
        default=2.5,
        help="Positive-class weight for source BCE. Use about neg/pos ratio to reduce conservative active logits.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="Patience for source Phase1/Phase2/Phase2.5 early stopping. 0 disables it.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0005,
        help="Minimum validation score improvement for source early stopping.",
    )
    parser.add_argument(
        "--early-stop-metric",
        type=str,
        default="combo",
        choices=["subset_acc", "element_acc", "loss", "combo"],
        help="Validation metric used for source early stopping.",
    )
    parser.add_argument("--da-pseudo-weight", type=float, default=0.02)
    parser.add_argument(
        "--da-pseudo-active-threshold",
        type=float,
        default=0.9,
        help="Only target subbands above this probability are pseudo-labeled active.",
    )
    parser.add_argument(
        "--da-pseudo-inactive-threshold",
        type=float,
        default=0.1,
        help="Only target subbands below this probability are pseudo-labeled inactive.",
    )
    parser.add_argument(
        "--da-pseudo-min-confidence",
        type=float,
        default=0.6,
        help="Minimum max(p,1-p) confidence for target pseudo-label/prototype terms.",
    )
    parser.add_argument(
        "--da-pseudo-gate-mode",
        type=str,
        default="fixed",
        choices=["fixed", "snr_quantile"],
        help="Pseudo-label gate. fixed uses explicit thresholds; snr_quantile estimates thresholds per target SNR bin.",
    )
    parser.add_argument(
        "--da-pseudo-quantile-rho",
        type=float,
        default=0.25,
        help="Tail ratio used by snr_quantile pseudo gate, e.g. 0.25 selects bottom/top 25%% per SNR bin.",
    )
    parser.add_argument(
        "--da-pseudo-quantile-update",
        type=str,
        default="frozen",
        choices=["frozen", "epoch"],
        help="Update schedule for snr_quantile pseudo gate. frozen estimates once before Phase3; epoch recomputes each epoch.",
    )
    parser.add_argument(
        "--da-pseudo-quantile-active-floor",
        type=float,
        default=0.5,
        help="Lower bound for snr_quantile active thresholds.",
    )
    parser.add_argument(
        "--da-pseudo-quantile-inactive-ceiling",
        type=float,
        default=0.05,
        help="Upper bound for snr_quantile inactive thresholds.",
    )
    parser.add_argument(
        "--prototype-weight",
        type=float,
        default=0.05,
        help="Weight for subband active/inactive source-prototype alignment.",
    )
    parser.add_argument(
        "--source-anchor-weight",
        type=float,
        default=0.1,
        help="Weight for preserving source detector behavior while adapting the target encoder.",
    )
    parser.add_argument(
        "--train-shared-projector",
        action="store_true",
        help="Also update the shared feature projector during occupancy-semantic DA.",
    )
    parser.add_argument("--phase3-init-domain-weight", type=float, default=0.005)
    parser.add_argument("--phase3-max-domain-weight", type=float, default=0.05)
    parser.add_argument(
        "--reuse-source-checkpoint",
        action="store_true",
        help="Reuse the latest matching source-only checkpoint under save-root. By default DA mode trains source/base from scratch.",
    )
    return parser.parse_args()


def parse_encoder_dilations(value):
    parts = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError("--encoder-dilations must contain exactly three comma-separated integers.")
    dilations = tuple(int(item) for item in parts)
    if any(item < 1 for item in dilations):
        raise ValueError("--encoder-dilations values must be positive integers.")
    return dilations


def _safe_path_token(value):
    token = os.path.splitext(os.path.basename(str(value)))[0]
    token = token.replace("radioml_wideband_multi_label_", "")
    token = token.replace("rician_", "Rician_")
    token = token.replace("rayleigh_", "Rayleigh_")
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in token)


def resolve_target_path(args):
    if args.target_path:
        return args.target_path
    target_key = f"{args.target_family.upper()}_TARGETS"
    return CONFIG[target_key][args.target_domain]


def resolve_source_path(args):
    if args.source_path:
        return args.source_path
    if args.source_family == "gaussian":
        return CONFIG["GAUSSIAN_SOURCE_PATH"]
    source_key = f"{args.source_family.upper()}_SOURCES"
    return CONFIG[source_key][args.source_domain]


def build_save_dirs(root, mode, model_tag=CONFIG["MODEL_TAG"], source_tag=None, target_tag=None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if mode == "basic":
        source_suffix = f"_{_safe_path_token(source_tag)}" if source_tag else ""
        folder = f"SourceOnly_{model_tag}{source_suffix}_{timestamp}"
    else:
        source_suffix = f"_{_safe_path_token(source_tag)}" if source_tag else ""
        target_suffix = f"_{_safe_path_token(target_tag)}" if target_tag else ""
        folder = f"Adapt_Mode_{model_tag}{source_suffix}_to{target_suffix}_{timestamp}"
    root = os.path.join(root, folder)
    checkpoints = os.path.join(root, "checkpoints")
    results = os.path.join(root, "results")
    os.makedirs(checkpoints, exist_ok=True)
    os.makedirs(results, exist_ok=True)
    return checkpoints, results


def save_loss_log(log_list, phase_name, save_path):
    csv_path = os.path.join(save_path, f"{phase_name}_loss.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if log_list and phase_name in {"phase1_source_cls", "phase2_cls", "phase2_finetune"}:
            if len(log_list[0]) >= 4:
                writer.writerow([
                    "Epoch",
                    "Average_Loss",
                    "Val_Element_Acc",
                    "Val_Subset_Acc",
                ])
            else:
                writer.writerow([
                    "Epoch",
                    "Average_Loss",
                ])
            writer.writerows(log_list)
        elif phase_name == "phase3_da" and log_list and len(log_list[0]) >= 6:
            if len(log_list[0]) >= 12:
                header = [
                    "Epoch",
                    "Average_Loss",
                    "Source_Anchor_Loss",
                    "Domain_Loss",
                    "Pseudo_Loss",
                    "Prototype_Loss",
                    "Pseudo_Coverage",
                    "Pseudo_Active_Coverage",
                    "Pseudo_Inactive_Coverage",
                    "Prototype_Active_Pairs",
                    "Prototype_Inactive_Pairs",
                    "GRL_Lambda",
                    "Domain_Weight",
                ]
                if len(log_list[0]) >= 14:
                    header.extend([
                        "Val_Element_Acc",
                        "Val_Subset_Acc",
                    ])
            else:
                header = [
                    "Epoch",
                    "Average_Loss",
                    "Cls_Loss",
                    "Domain_Loss",
                    "Global_Domain_Loss",
                    "Pseudo_Loss",
                    "GRL_Lambda",
                    "Domain_Weight",
                ]
            writer.writerow(header)
            writer.writerows(log_list)
        else:
            writer.writerow(["Epoch", "Average_Loss"])
            for epoch, loss in log_list:
                writer.writerow([epoch, loss])


def save_train_summary(info, save_path):
    txt_path = os.path.join(save_path, "train_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("Training Summary\n")
        f.write("=" * 50 + "\n")
        for key, value in info.items():
            f.write(f"{key}: {value}\n")


def load_checkpoint(model, checkpoint_path, device, checkpoint_name):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{checkpoint_name} not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    result = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded {checkpoint_name}: {checkpoint_path}", flush=True)
    if result.missing_keys:
        print(f"Missing keys: {list(result.missing_keys)}", flush=True)
    if result.unexpected_keys:
        print(f"Unexpected keys: {list(result.unexpected_keys)}", flush=True)


def load_existing_basic_checkpoint(model, save_root, device, model_tag, source_tag=None, force_from_scratch=False):
    if force_from_scratch:
        return False, None

    candidates = []
    pattern_prefix = f"SourceOnly_{model_tag}"
    normalized_source_tag = _safe_path_token(source_tag) if source_tag else None

    try:
        subdirs = [
            entry for entry in os.scandir(save_root)
            if entry.is_dir() and entry.name.startswith(pattern_prefix)
        ]
        if normalized_source_tag is not None:
            subdirs = [
                entry for entry in subdirs
                if f"_{normalized_source_tag}_" in entry.name
            ]
        subdirs.sort(key=lambda e: e.stat().st_mtime, reverse=True)
        for entry in subdirs:
            candidates.append(os.path.join(entry.path, "checkpoints", "phase2_finetuned.pth"))
        for entry in subdirs:
            candidates.append(os.path.join(entry.path, "checkpoints", "phase2_cls_trained.pth"))
    except FileNotFoundError:
        pass

    for checkpoint_path in candidates:
        if os.path.exists(checkpoint_path):
            load_checkpoint(model, checkpoint_path, device, "source-only checkpoint")
            return True, checkpoint_path
    return False, None


def build_source_loaders(source_path, modulation, batch_size, train_snr_list=None, normalize_iq=True, seed=CONFIG["SEED"]):
    source_data, source_labels, source_snr = load_source_data(
        source_path,
        modulation,
        snr_list=train_snr_list,
        normalize_iq=normalize_iq,
    )
    print(f"Source loaded | samples: {source_data.shape[0]}", flush=True)
    x_train, x_val, x_test, y_train, y_val, y_test, snr_tr, snr_val, snr_te = split_source_train_val_test(
        source_data,
        source_labels,
        source_snr,
        seed=seed,
    )
    source_dataset = TensorDataset(x_train, y_train, torch.tensor(snr_tr, device=x_train.device))
    train_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )
    val_loader = build_dataloader(x_val, y_val, batch_size, shuffle=False, drop_last=False)
    test_loader = build_dataloader(x_test, y_test, batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader


def build_target_loader(target_path, modulation, batch_size, train_snr_list=None, seed=CONFIG["SEED"], normalize_iq=True):
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Target data file not found: {target_path}")

    # Use the same deterministic split as osada.evaluation, but discard labels
    # for DA. This keeps target-test IQ unseen during adaptation.
    set_random_seed(seed)
    target_data, target_labels, target_snr = load_source_data(
        target_path,
        modulation,
        snr_list=train_snr_list,
        normalize_iq=normalize_iq,
    )
    x_train, x_val, x_test, _, _, _, snr_train, _, _ = split_source_train_val_test(
        target_data,
        target_labels,
        target_snr,
        seed=seed,
    )
    print(
        "Target loaded | "
        f"total: {target_data.shape[0]} | "
        f"unlabeled_train: {x_train.shape[0]} | "
        f"heldout_val: {x_val.shape[0]} | "
        f"heldout_test: {x_test.shape[0]}",
        flush=True,
    )
    target_dataset = TensorDataset(x_train, torch.tensor(snr_train, device=x_train.device))
    return DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )


def main():
    args = parse_args()
    mode = args.mode
    train_snr_list = CONFIG["TRAIN_SNR_LIST"]
    start_time = time.time()
    source_path = resolve_source_path(args)
    target_path = resolve_target_path(args) if mode == "da" else None

    checkpoint_dir, result_dir = build_save_dirs(
        args.save_root,
        mode,
        args.model_tag,
        source_tag=f"{args.source_family}_{args.source_domain}" if args.source_family != "gaussian" else "gaussian",
        target_tag=f"{args.target_family}_{args.target_domain}" if mode == "da" else None,
    )

    print("=" * 70, flush=True)
    print("Wideband MR-CDA training started", flush=True)
    print(f"Model: {args.model_tag}", flush=True)
    print(f"Mode: {mode}", flush=True)
    print(f"Training SNR list: {train_snr_list}", flush=True)
    print(f"FFT norm mode: {args.fft_norm_mode}", flush=True)
    print(f"Input unit-energy normalization: {not args.no_input_unit_energy}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"Checkpoint dir: {checkpoint_dir}", flush=True)
    print("=" * 70, flush=True)

    set_random_seed(args.seed)

    print("\nLoading source domain data...", flush=True)
    source_loader, source_val_loader, source_test_loader = build_source_loaders(
        source_path,
        CONFIG["MODULATION"],
        args.batch_size,
        train_snr_list=train_snr_list,
        normalize_iq=not args.no_input_unit_energy,
        seed=args.seed,
    )

    model = BroadBandDomainAdaptNet(
        num_subbands=8,
        num_domains=2,
        fft_norm_mode=args.fft_norm_mode,
        encoder_norm=args.encoder_norm,
        encoder_gn_groups=args.encoder_gn_groups,
        encoder_dilations=parse_encoder_dilations(args.encoder_dilations),
        encoder_use_se=args.encoder_use_se,
        encoder_multiscale=args.encoder_multiscale,
    ).to(DEVICE)
    print("Model initialized", flush=True)
    print(
        f"Backbone config | norm:{args.encoder_norm} | gn_groups:{args.encoder_gn_groups} | dilations:{args.encoder_dilations} | "
        f"SE:{args.encoder_use_se} | multiscale:{args.encoder_multiscale}",
        flush=True,
    )

    loaded_basic = False
    loaded_checkpoint = None

    if mode == "basic":
        model, phase1_log = train_phase1_source_cls(
            model,
            source_loader,
            source_val_loader,
            DEVICE,
            args.epochs_phase1,
            args.lr_phase1,
            checkpoint_dir,
            aux_weight=args.source_aux_weight,
            subband_reweight_mode=args.source_subband_reweight_mode,
            source_pos_weight=args.source_pos_weight,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            early_stop_metric=args.early_stop_metric,
        )
        save_loss_log(phase1_log, "phase1_source_cls", result_dir)

        model, phase2_log = train_phase2_cls(
            model,
            source_loader,
            source_val_loader,
            DEVICE,
            args.epochs_phase2,
            args.lr_phase2,
            checkpoint_dir,
            aux_weight=args.source_aux_weight,
            subband_reweight_mode=args.source_subband_reweight_mode,
            source_pos_weight=args.source_pos_weight,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            early_stop_metric=args.early_stop_metric,
        )
        save_loss_log(phase2_log, "phase2_cls", result_dir)

        model, phase2_finetune_log = train_phase2_finetune(
            model,
            source_loader,
            source_val_loader,
            DEVICE,
            args.epochs_phase2_finetune,
            args.lr_finetune_encoder,
            args.lr_finetune_classifier,
            checkpoint_dir,
            unfreeze_last_blocks=args.finetune_unfreeze_last_blocks,
            aux_weight=args.source_aux_weight,
            subband_reweight_mode=args.source_subband_reweight_mode,
            source_pos_weight=args.source_pos_weight,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            early_stop_metric=args.early_stop_metric,
        )
        save_loss_log(phase2_finetune_log, "phase2_finetune", result_dir)

    elif mode == "da":
        print("\nLoading target domain data...", flush=True)
        target_loader = build_target_loader(
            target_path,
            CONFIG["MODULATION"],
            args.batch_size,
            train_snr_list=train_snr_list,
            seed=args.seed,
            normalize_iq=not args.no_input_unit_energy,
        )

        if args.base_checkpoint:
            load_checkpoint(model, args.base_checkpoint, DEVICE, "base source-only checkpoint")
            loaded_basic = True
            loaded_checkpoint = args.base_checkpoint
        else:
            loaded_basic, loaded_checkpoint = load_existing_basic_checkpoint(
                model,
                args.save_root,
                DEVICE,
                args.model_tag,
                source_tag=f"{args.source_family}_{args.source_domain}" if args.source_family != "gaussian" else "gaussian",
                force_from_scratch=not args.reuse_source_checkpoint,
            )

        if not loaded_basic:
            print("\nNo source-only checkpoint found. Running source-only stages first.", flush=True)
            model, phase1_log = train_phase1_source_cls(
                model,
                source_loader,
                source_val_loader,
                DEVICE,
                args.epochs_phase1,
                args.lr_phase1,
                checkpoint_dir,
                aux_weight=args.source_aux_weight,
                subband_reweight_mode=args.source_subband_reweight_mode,
                source_pos_weight=args.source_pos_weight,
                early_stop_patience=args.early_stop_patience,
                early_stop_min_delta=args.early_stop_min_delta,
                early_stop_metric=args.early_stop_metric,
            )
            save_loss_log(phase1_log, "phase1_source_cls", result_dir)

            model, phase2_log = train_phase2_cls(
                model,
                source_loader,
                source_val_loader,
                DEVICE,
                args.epochs_phase2,
                args.lr_phase2,
                checkpoint_dir,
                aux_weight=args.source_aux_weight,
                subband_reweight_mode=args.source_subband_reweight_mode,
                source_pos_weight=args.source_pos_weight,
                early_stop_patience=args.early_stop_patience,
                early_stop_min_delta=args.early_stop_min_delta,
                early_stop_metric=args.early_stop_metric,
            )
            save_loss_log(phase2_log, "phase2_cls", result_dir)

            model, phase2_finetune_log = train_phase2_finetune(
                model,
                source_loader,
                source_val_loader,
                DEVICE,
                args.epochs_phase2_finetune,
                args.lr_finetune_encoder,
                args.lr_finetune_classifier,
                checkpoint_dir,
                unfreeze_last_blocks=args.finetune_unfreeze_last_blocks,
                aux_weight=args.source_aux_weight,
                subband_reweight_mode=args.source_subband_reweight_mode,
                source_pos_weight=args.source_pos_weight,
                early_stop_patience=args.early_stop_patience,
                early_stop_min_delta=args.early_stop_min_delta,
                early_stop_metric=args.early_stop_metric,
            )
            save_loss_log(phase2_finetune_log, "phase2_finetune", result_dir)

        model, phase3_log = train_phase3_da(
            model,
            source_loader,
            target_loader,
            source_val_loader,
            DEVICE,
            args.epochs_phase3,
            args.lr_phase3,
            checkpoint_dir,
            init_domain_weight=args.phase3_init_domain_weight,
            max_domain_weight=args.phase3_max_domain_weight,
            detach_condition=not args.da_no_detach_condition,
            subband_reweight_mode=args.da_subband_reweight_mode,
            aux_weight=args.source_aux_weight,
            pseudo_weight=args.da_pseudo_weight,
            pseudo_active_threshold=args.da_pseudo_active_threshold,
            pseudo_inactive_threshold=args.da_pseudo_inactive_threshold,
            pseudo_min_confidence=args.da_pseudo_min_confidence,
            pseudo_gate_mode=args.da_pseudo_gate_mode,
            pseudo_quantile_rho=args.da_pseudo_quantile_rho,
            pseudo_quantile_update=args.da_pseudo_quantile_update,
            pseudo_quantile_active_floor=args.da_pseudo_quantile_active_floor,
            pseudo_quantile_inactive_ceiling=args.da_pseudo_quantile_inactive_ceiling,
            prototype_weight=args.prototype_weight,
            source_anchor_weight=args.source_anchor_weight,
            train_shared_projector=args.train_shared_projector,
            use_occupancy_condition=not args.da_no_occupancy_condition,
        )
        save_loss_log(phase3_log, "phase3_da", result_dir)

    total_time = time.time() - start_time
    summary = {
        "model": args.model_tag,
        "seed": args.seed,
        "mode": mode,
        "source_family": args.source_family,
        "source_domain": args.source_domain if args.source_family != "gaussian" else "N/A",
        "source_path": source_path,
        "target_family": args.target_family if mode == "da" else "N/A",
        "target_domain": args.target_domain if mode == "da" else "N/A",
        "target_path": target_path if mode == "da" else "N/A",
        "target_unlabeled_split": "train_only_70pct_heldout_test_20pct" if mode == "da" else "N/A",
        "base_checkpoint": loaded_checkpoint if loaded_basic else "None",
        "phase3_init_domain_weight": args.phase3_init_domain_weight if mode == "da" else "N/A",
        "phase3_max_domain_weight": args.phase3_max_domain_weight if mode == "da" else "N/A",
        "da_use_occupancy_condition": (not args.da_no_occupancy_condition) if mode == "da" else "N/A",
        "da_detach_condition": (not args.da_no_detach_condition) if mode == "da" else "N/A",
        "da_subband_reweight_mode": args.da_subband_reweight_mode if mode == "da" else "N/A",
        "source_subband_reweight_mode": args.source_subband_reweight_mode,
        "fft_norm_mode": args.fft_norm_mode,
        "encoder_norm": args.encoder_norm,
        "encoder_gn_groups": args.encoder_gn_groups,
        "encoder_dilations": args.encoder_dilations,
        "encoder_use_se": args.encoder_use_se,
        "encoder_multiscale": args.encoder_multiscale,
        "input_unit_energy_normalization": not args.no_input_unit_energy,
        "source_aux_weight": args.source_aux_weight,
        "source_pos_weight": args.source_pos_weight,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "early_stop_metric": args.early_stop_metric,
        "da_pseudo_weight": args.da_pseudo_weight if mode == "da" else "N/A",
        "da_pseudo_active_threshold": args.da_pseudo_active_threshold if mode == "da" else "N/A",
        "da_pseudo_inactive_threshold": args.da_pseudo_inactive_threshold if mode == "da" else "N/A",
        "da_pseudo_min_confidence": args.da_pseudo_min_confidence if mode == "da" else "N/A",
        "da_pseudo_gate_mode": args.da_pseudo_gate_mode if mode == "da" else "N/A",
        "da_pseudo_quantile_rho": args.da_pseudo_quantile_rho if mode == "da" else "N/A",
        "da_pseudo_quantile_update": args.da_pseudo_quantile_update if mode == "da" else "N/A",
        "da_pseudo_quantile_active_floor": args.da_pseudo_quantile_active_floor if mode == "da" else "N/A",
        "da_pseudo_quantile_inactive_ceiling": args.da_pseudo_quantile_inactive_ceiling if mode == "da" else "N/A",
        "prototype_weight": args.prototype_weight if mode == "da" else "N/A",
        "source_anchor_weight": args.source_anchor_weight if mode == "da" else "N/A",
        "train_shared_projector": args.train_shared_projector if mode == "da" else "N/A",
        "total_time": f"{total_time:.1f}s",
        "task": "8-subband multi-label spectrum sensing",
        "train_snr_list": train_snr_list,
        "variant": "occupancy_semantic_adversarial_adaptation",
    }
    save_train_summary(summary, result_dir)


if __name__ == "__main__":
    main()


