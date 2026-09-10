#!/usr/bin/env bash
set -euo pipefail

DATE_TAG="20260730"
SEED="42"
RUN_SUFFIX="rician_seed42"
BATCH_SIZE="32"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --date-tag) DATE_TAG="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --run-suffix) RUN_SUFFIX="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash scripts/train_rician.sh [--date-tag YYYYMMDD] [--seed N] [--run-suffix NAME] [--batch-size N]"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

SAVE_ROOT="runs/rician_${DATE_TAG}_${RUN_SUFFIX}"
SRC="data/source_gaussian_qpsk_rxshift_100mhz.pkl"
TGT="data/target_rician_qpsk_rxshift_100mhz.pkl"

[[ -f "$SRC" ]] || { echo "Source data file not found: $SRC" >&2; exit 1; }
[[ -f "$TGT" ]] || { echo "Rician target data file not found: $TGT" >&2; exit 1; }
mkdir -p "$SAVE_ROOT"

run_cmd() {
    local name="$1"
    shift
    echo
    echo "===== $name ====="
    "$@"
}

latest_run_dir() {
    local pattern="$1"
    local run_dir
    run_dir="$(find "$SAVE_ROOT" -maxdepth 1 -type d -name "$pattern" -printf '%T@ %p\n' | sort -nr | head -n 1 | awk '{$1=""; sub(/^ /, ""); print}')"
    [[ -n "$run_dir" ]] || { echo "No run directory found for pattern: $pattern" >&2; exit 1; }
    echo "$run_dir"
}

phase2_ckpt() { echo "$(latest_run_dir "$1")/checkpoints/phase2_finetuned.pth"; }
final_ckpt() { echo "$(latest_run_dir "$1")/checkpoints/final_occupancy_semantic_adapt_model.pth"; }

COMMON_SOURCE_ARGS=(
    -m osada.train --mode basic --source-family gaussian --source-path "$SRC"
    --save-root "$SAVE_ROOT" --seed "$SEED" --batch-size "$BATCH_SIZE"
    --fft-norm-mode log_power --encoder-norm bn --encoder-dilations 1,1,1 --encoder-use-se
    --source-subband-reweight-mode soft --source-aux-weight 0.3 --source-pos-weight 2.5
    --early-stop-patience 5 --early-stop-min-delta 0.0005 --early-stop-metric combo
)

COMMON_DA_ARGS=(
    -m osada.train --mode da --source-family gaussian --source-path "$SRC"
    --target-family rician --target-domain hard --target-path "$TGT"
    --save-root "$SAVE_ROOT" --seed "$SEED" --batch-size "$BATCH_SIZE"
    --fft-norm-mode log_power --encoder-norm bn --encoder-dilations 1,1,1 --encoder-use-se
    --source-subband-reweight-mode soft --da-subband-reweight-mode soft
    --source-aux-weight 0.3 --source-pos-weight 2.5
    --early-stop-patience 5 --early-stop-min-delta 0.0005 --early-stop-metric combo
    --phase3-init-domain-weight 0.005 --phase3-max-domain-weight 0.05 --source-anchor-weight 0.1
)

run_cmd "01 source-only" python "${COMMON_SOURCE_ARGS[@]}" --model-tag "source_only_${DATE_TAG}"
SOURCE_CKPT="$(phase2_ckpt "SourceOnly_source_only_${DATE_TAG}*")"

run_cmd "02 Full OSADA main" python "${COMMON_DA_ARGS[@]}" --model-tag "main_full_osada_${DATE_TAG}" --base-checkpoint "$SOURCE_CKPT" --da-pseudo-weight 0.02 --da-pseudo-gate-mode snr_quantile --prototype-weight 0.05
FULL_CKPT="$(final_ckpt "Adapt_Mode_main_full_osada_${DATE_TAG}*")"

run_cmd "03 DA-Occ ablation" python "${COMMON_DA_ARGS[@]}" --model-tag "ablation_02_occupancy_da_${DATE_TAG}" --base-checkpoint "$SOURCE_CKPT" --da-pseudo-weight 0.0 --da-pseudo-gate-mode fixed --prototype-weight 0.0
run_cmd "04 Occ+Pseudo ablation" python "${COMMON_DA_ARGS[@]}" --model-tag "ablation_03_occupancy_pseudo_${DATE_TAG}" --base-checkpoint "$SOURCE_CKPT" --da-pseudo-weight 0.02 --da-pseudo-gate-mode snr_quantile --prototype-weight 0.0

OCC_CKPT="$(final_ckpt "Adapt_Mode_ablation_02_occupancy_da_${DATE_TAG}*")"
OCC_PSEUDO_CKPT="$(final_ckpt "Adapt_Mode_ablation_03_occupancy_pseudo_${DATE_TAG}*")"

run_cmd "05 print main evaluation" python evaluate_main.py --eval-domains rician --source-checkpoint "$SOURCE_CKPT" --osada-checkpoint "$FULL_CKPT" --rician-path "$TGT" --batch-size "$BATCH_SIZE" --no-print-per-snr
run_cmd "06 print ablation evaluation" python evaluate_ablation.py --target-path "$TGT" --domain-name Rician --da-occ-checkpoint "$OCC_CKPT" --occ-pseudo-checkpoint "$OCC_PSEUDO_CKPT" --full-osada-checkpoint "$FULL_CKPT" --batch-size "$BATCH_SIZE" --no-print-per-snr

echo
echo "Done. Run folder: $SAVE_ROOT"
