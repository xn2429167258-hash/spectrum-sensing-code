import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def parse_snr_list(value):
    if value is None or str(value).strip() == "":
        return TRAIN_SNR_LIST
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_encoder_dilations(value):
    parts = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError("--encoder-dilations must contain exactly three comma-separated integers.")
    dilations = tuple(int(item) for item in parts)
    if any(item < 1 for item in dilations):
        raise ValueError("--encoder-dilations values must be positive integers.")
    return dilations


def binary_counts(y_true, y_pred):
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fn, fp, tn


def pd_from_counts(tp, fn, fp, tn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def run_inference(model, loader, device, subband_reweight_mode="soft"):
    all_probs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device, non_blocking=True)
            logits, _, _ = model(
                x,
                grl_lambda=0.0,
                use_conditional_domain=False,
                subband_reweight_mode=subband_reweight_mode,
            )
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy().astype(np.int32))

    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def predict_with_thresholds(probs, thresholds):
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if thresholds.ndim == 0:
        return (probs >= float(thresholds)).astype(np.int32)
    return (probs >= thresholds.reshape(1, -1)).astype(np.int32)


def compute_overall_metrics(labels, preds):
    subset_acc = accuracy_score(labels, preds)
    subband_acc = float(np.mean(labels == preds))
    precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
    precision_micro = precision_score(labels, preds, average="micro", zero_division=0)
    pd_macro = recall_score(labels, preds, average="macro", zero_division=0)
    pd_micro = recall_score(labels, preds, average="micro", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)

    return {
        "subset_accuracy_exact_match": subset_acc,
        "subband_accuracy_element_level": subband_acc,
        "precision_macro": precision_macro,
        "precision_micro": precision_micro,
        "pd_macro_recall": pd_macro,
        "pd_micro_recall": pd_micro,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
    }


def compute_per_snr_metrics(labels, preds, snrs):
    rows = []
    for snr in np.sort(np.unique(snrs)):
        mask = snrs == snr
        labels_snr = labels[mask]
        preds_snr = preds[mask]
        if labels_snr.size == 0:
            continue

        pd_list = []
        f1_list = []
        for band_idx in range(labels.shape[1]):
            y_true = labels_snr[:, band_idx]
            y_pred = preds_snr[:, band_idx]
            tp, fn, fp, tn = binary_counts(y_true, y_pred)
            pd = pd_from_counts(tp, fn, fp, tn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * precision * pd / (precision + pd) if (precision + pd) > 0 else 0.0
            pd_list.append(pd)
            f1_list.append(f1)

        rows.append(
            {
                "snr": int(snr),
                "pd_macro": float(np.mean(pd_list)),
                "f1_macro": float(np.mean(f1_list)),
                "element_acc": float(np.mean(preds_snr == labels_snr)),
                "subset_acc": float(np.mean(np.all(preds_snr == labels_snr, axis=1))),
            }
        )
    return rows
