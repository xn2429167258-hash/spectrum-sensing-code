import copy
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim

from osada.losses import (
    classification_loss,
    conservative_subband_pseudo_loss,
    subband_occupancy_prototype_loss,
)


"""
Training utilities for the clean frequency-only Model3 variant.

All supervised losses, validation metrics, and DA alignment terms operate on
the frequency detector logits. The old time/fusion branches are intentionally
absent from this file.
"""


def _to_device(x, device):
    return x.to(device, non_blocking=True)


def _unpack_source_batch(batch, device):
    x, labels = batch[0], batch[1]
    return _to_device(x, device), _to_device(labels, device)


def _unpack_source_batch_with_snr(batch, device):
    x, labels = _unpack_source_batch(batch, device)
    snr = None
    if isinstance(batch, (tuple, list)) and len(batch) >= 3:
        snr = _to_device(batch[2], device)
    return x, labels, snr


def _build_sample_weight(snr, low_snr_threshold=None, low_snr_weight=1.0):
    if snr is None or low_snr_threshold is None or low_snr_weight == 1.0:
        return None
    weights = torch.ones_like(snr, dtype=torch.float32).to(snr.device)
    weights[snr <= low_snr_threshold] = float(low_snr_weight)
    return weights


def _unpack_target_batch(batch, device):
    if isinstance(batch, (tuple, list)):
        x = batch[0]
    else:
        x = batch
    return _to_device(x, device)


def _unpack_target_batch_with_snr(batch, device):
    if isinstance(batch, (tuple, list)):
        x = batch[0]
        snr = batch[1] if len(batch) >= 2 else None
    else:
        x = batch
        snr = None
    x = _to_device(x, device)
    if snr is not None:
        snr = _to_device(snr, device)
    return x, snr


def _make_threshold_matrix(target_snr, threshold_table, fallback_active, fallback_inactive, num_subbands, device):
    if target_snr is None or threshold_table is None:
        return fallback_active, fallback_inactive

    active_threshold = torch.full(
        (target_snr.size(0), int(num_subbands)),
        float(fallback_active),
        device=device,
        dtype=torch.float32,
    )
    inactive_threshold = torch.full_like(active_threshold, float(fallback_inactive))
    snr_cpu = target_snr.detach().cpu().view(-1).tolist()
    for row_idx, snr_value in enumerate(snr_cpu):
        key = int(snr_value)
        if key not in threshold_table:
            continue
        active_value, inactive_value = threshold_table[key]
        active_threshold[row_idx].fill_(float(active_value))
        inactive_threshold[row_idx].fill_(float(inactive_value))
    return active_threshold, inactive_threshold


def _estimate_snr_quantile_thresholds(
    model,
    target_loader,
    device,
    rho=0.25,
    num_subbands=8,
    subband_reweight_mode="soft",
    fallback_active=0.9,
    fallback_inactive=0.1,
    active_floor=None,
    inactive_ceiling=None,
):
    rho = float(rho)
    if not (0.0 < rho < 0.5):
        raise ValueError("--da-pseudo-quantile-rho must be in (0, 0.5).")

    was_training = model.training
    model.eval()
    probs_by_snr = {}
    with torch.no_grad():
        for batch in target_loader:
            x_t, snr_t = _unpack_target_batch_with_snr(batch, device)
            if snr_t is None:
                continue
            logits_t, _, _ = _forward_model(
                model,
                x_t,
                grl_lambda=0.0,
                use_conditional_domain=False,
                subband_reweight_mode=subband_reweight_mode,
            )
            probs = torch.sigmoid(logits_t).detach().cpu()
            snrs = snr_t.detach().cpu().view(-1)
            for snr_value in torch.unique(snrs):
                mask = snrs == snr_value
                probs_by_snr.setdefault(int(snr_value.item()), []).append(probs[mask].reshape(-1))

    if was_training:
        model.train()

    table = {}
    for snr_value, chunks in probs_by_snr.items():
        values = torch.cat(chunks)
        if values.numel() == 0:
            continue
        inactive_value = torch.quantile(values, rho).item()
        active_value = torch.quantile(values, 1.0 - rho).item()
        if not math.isfinite(active_value):
            active_value = float(fallback_active)
        if not math.isfinite(inactive_value):
            inactive_value = float(fallback_inactive)
        if active_floor is not None:
            active_value = max(float(active_value), float(active_floor))
        if inactive_ceiling is not None:
            inactive_value = min(float(inactive_value), float(inactive_ceiling))
        table[int(snr_value)] = (float(active_value), float(inactive_value))
    return table


def _format_threshold_table(threshold_table):
    if not threshold_table:
        return "SNR bins:0"
    active_values = [item[0] for item in threshold_table.values()]
    inactive_values = [item[1] for item in threshold_table.values()]
    return (
        f"SNR bins:{len(threshold_table)} | "
        f"tau_a:{min(active_values):.4f}-{max(active_values):.4f} | "
        f"tau_i:{min(inactive_values):.4f}-{max(inactive_values):.4f}"
    )


def _forward_model(
    model,
    x,
    subband_reweight_mode="soft",
    return_features=False,
    grl_lambda=None,
    use_conditional_domain=None,
    **unused_da_kwargs,
):
    return model(
        x,
        subband_reweight_mode=subband_reweight_mode,
        return_features=return_features,
    )


def _grl_schedule(epoch, epochs, warmup_epochs=0, warmup_value=0.0):
    if epochs <= 1:
        return 1.0
    warmup_epochs = max(0, min(int(warmup_epochs), max(epochs - 1, 0)))
    if epoch < warmup_epochs:
        return float(warmup_value)
    remaining_epochs = epochs - warmup_epochs
    if remaining_epochs <= 1:
        return 1.0
    progress = (epoch - warmup_epochs) / float(remaining_epochs - 1)
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


def _linear_schedule(epoch, epochs, start, end, warmup_epochs=0, warmup_value=None):
    if epochs <= 1:
        return end
    warmup_epochs = max(0, min(int(warmup_epochs), max(epochs - 1, 0)))
    if warmup_value is None:
        warmup_value = start
    if epoch < warmup_epochs:
        return float(warmup_value)
    remaining_epochs = epochs - warmup_epochs
    if remaining_epochs <= 1:
        return end
    progress = (epoch - warmup_epochs) / float(remaining_epochs - 1)
    return start + progress * (end - start)


def _ensure_save_dir(save_dir):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)


class OccupancyDomainDiscriminator(nn.Module):
    def __init__(self, feature_dim, num_subbands, hidden_dim=128, use_occupancy_condition=True):
        super().__init__()
        self.use_occupancy_condition = bool(use_occupancy_condition)
        input_dim = feature_dim + num_subbands if self.use_occupancy_condition else feature_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, shared_feature, occupancy_condition, grl_lambda=None):
        if self.use_occupancy_condition:
            if occupancy_condition is None:
                raise ValueError("occupancy_condition is required when use_occupancy_condition=True")
            x = torch.cat([shared_feature, occupancy_condition], dim=1)
        else:
            x = shared_feature
        if grl_lambda is not None:
            x = GRL.apply(x, grl_lambda)
        return self.net(x)


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_grl=1.0):
        if torch.is_tensor(lambda_grl):
            lambda_grl = lambda_grl.to(device=x.device, dtype=x.dtype)
        else:
            lambda_grl = torch.tensor(float(lambda_grl), device=x.device, dtype=x.dtype)
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_grl = ctx.lambda_grl
        while lambda_grl.dim() < grad_output.dim():
            lambda_grl = lambda_grl.unsqueeze(-1)
        return -lambda_grl * grad_output, None


def _freeze_all(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def _configure_target_encoder_adaptation(model, train_shared_projector=False):
    for param in model.parameters():
        param.requires_grad = False

    for encoder in _frequency_encoders(model):
        for param in encoder.parameters():
            param.requires_grad = True

    if train_shared_projector and hasattr(model, "shared_projector"):
        for param in model.shared_projector.parameters():
            param.requires_grad = True


def _set_frozen_detector_eval(model):
    for module_name in ("freq_classifier", "subband_reweight"):
        if hasattr(model, module_name):
            getattr(model, module_name).eval()
    if hasattr(model, "shared_projector"):
        model.shared_projector.eval()


def _set_trainable(model, train_encoder=True, train_domain=False):
    for name, param in model.named_parameters():
        if ("domain_disc" in name) or ("shared_projector" in name):
            param.requires_grad = train_domain
        elif "freq_encoder" in name:
            param.requires_grad = train_encoder
        else:
            param.requires_grad = True


def _unfreeze_last_encoder_blocks(encoder, unfreeze_last_blocks):
    block_count = max(1, int(unfreeze_last_blocks))
    candidate_nets = [encoder.net] if hasattr(encoder, "net") else [encoder]

    for net in candidate_nets:
        children = list(net.children())
        modules = children[-block_count:] if children else [net]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True


def _frequency_encoders(model):
    encoders = []
    for attr_name in ("raw_freq_encoder", "contrast_freq_encoder"):
        if hasattr(model, attr_name):
            encoders.append(getattr(model, attr_name))
    if encoders:
        return encoders
    if hasattr(model, "freq_encoder"):
        return [model.freq_encoder]
    return []


def _compute_accuracy(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(dtype=labels.dtype)
    element_correct = (preds == labels).float()
    subset_correct = element_correct.all(dim=1).float()
    return float(element_correct.mean().item()), float(subset_correct.mean().item())


def evaluate_multiview_classification(model, loader, device, aux_weight=0.3, subband_reweight_mode="soft"):
    model.eval()
    total = {
        "loss": 0.0,
        "element_acc": 0.0,
        "subset_acc": 0.0,
        "freq_loss": 0.0,
        "freq_element_acc": 0.0,
        "freq_subset_acc": 0.0,
    }
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            x, labels = _unpack_source_batch(batch, device)
            logits, _, _ = _forward_model(
                model,
                x,
                grl_lambda=0.0,
                use_conditional_domain=False,
                subband_reweight_mode=subband_reweight_mode,
            )
            loss = classification_loss(logits, labels)
            element_acc, subset_acc = _compute_accuracy(logits, labels)

            total["loss"] += loss.item()
            total["element_acc"] += element_acc
            total["subset_acc"] += subset_acc
            total["freq_loss"] += loss.item()
            total["freq_element_acc"] += element_acc
            total["freq_subset_acc"] += subset_acc
            num_batches += 1

    denom = max(num_batches, 1)
    return {key: value / denom for key, value in total.items()}


def _format_val_stats(val_stats):
    return (
        f"val_elem:{val_stats['element_acc']:.4f} "
        f"val_subset:{val_stats['subset_acc']:.4f}"
    )


def _make_pos_weight(value, logits):
    if value is None or float(value) <= 0.0 or abs(float(value) - 1.0) < 1e-8:
        return None
    return torch.full(
        (logits.size(1),),
        float(value),
        device=logits.device,
        dtype=logits.dtype,
    )


def _early_stop_score(val_stats, metric):
    if metric == "loss":
        return -float(val_stats["loss"])
    if metric == "element_acc":
        return float(val_stats["element_acc"])
    if metric == "subset_acc":
        return float(val_stats["subset_acc"])
    if metric == "combo":
        return float(val_stats["subset_acc"]) + 0.1 * float(val_stats["element_acc"])
    raise ValueError(f"Unsupported early_stop_metric: {metric}")


def _copy_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _train_source_epoch(
    model,
    source_loader,
    optimizer,
    device,
    subband_reweight_mode="soft",
    low_snr_threshold=None,
    low_snr_weight=1.0,
    cls_pos_weight=None,
):
    model.train()
    total_loss = 0.0

    for batch in source_loader:
        x, labels, snr = _unpack_source_batch_with_snr(batch, device)
        optimizer.zero_grad()
        logits, _, _ = _forward_model(
            model,
            x,
            grl_lambda=0.0,
            use_conditional_domain=False,
            subband_reweight_mode=subband_reweight_mode,
        )
        sample_weight = _build_sample_weight(snr, low_snr_threshold, low_snr_weight)
        pos_weight = _make_pos_weight(cls_pos_weight, logits)
        loss = classification_loss(logits, labels, pos_weight=pos_weight, sample_weight=sample_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(source_loader), 1)


def train_phase1_source_cls(
    model,
    source_loader,
    val_loader=None,
    device=None,
    epochs=30,
    lr=1e-3,
    save_dir=None,
    aux_weight=0.3,
    subband_reweight_mode="soft",
    source_pos_weight=1.0,
    early_stop_patience=0,
    early_stop_min_delta=0.0,
    early_stop_metric="subset_acc",
):
    _ensure_save_dir(save_dir)
    device = device or next(model.parameters()).device
    _set_trainable(model, train_encoder=True, train_domain=False)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    log = []
    best_score = None
    best_state = None
    stale_epochs = 0

    print("\n" + "=" * 50)
    print("Phase1: source frequency detector pretraining")
    print(
        f"source_pos_weight={source_pos_weight} | "
        f"early_stop_patience={early_stop_patience} | "
        f"early_stop_metric={early_stop_metric}",
        flush=True,
    )
    print("=" * 50)

    for epoch in range(epochs):
        avg_loss = _train_source_epoch(
            model,
            source_loader,
            optimizer,
            device,
            subband_reweight_mode=subband_reweight_mode,
            cls_pos_weight=source_pos_weight,
        )

        if val_loader is not None:
            val_stats = evaluate_multiview_classification(
                model,
                val_loader,
                device,
                subband_reweight_mode=subband_reweight_mode,
            )
            log.append([
                epoch + 1,
                avg_loss,
                val_stats["element_acc"],
                val_stats["subset_acc"],
            ])
            print(
                f"Phase1 | Epoch [{epoch + 1}/{epochs}] | "
                f"loss:{avg_loss:.4f} | {_format_val_stats(val_stats)}",
                flush=True,
            )
            if early_stop_patience > 0:
                score = _early_stop_score(val_stats, early_stop_metric)
                improved = best_score is None or score > best_score + float(early_stop_min_delta)
                if improved:
                    best_score = score
                    best_state = _copy_state_dict(model)
                    stale_epochs = 0
                    if save_dir is not None:
                        torch.save(model.state_dict(), os.path.join(save_dir, "phase1_source_pretrained_best.pth"))
                else:
                    stale_epochs += 1
                    if stale_epochs >= int(early_stop_patience):
                        print(
                            f"Phase1 early stopped at epoch {epoch + 1}; "
                            f"best_{early_stop_metric}={best_score:.6f}",
                            flush=True,
                        )
                        break
        else:
            log.append([epoch + 1, avg_loss])
            print(f"Phase1 | Epoch [{epoch + 1}/{epochs}] | loss:{avg_loss:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    if save_dir is not None:
        torch.save(model.state_dict(), os.path.join(save_dir, "phase1_source_pretrained.pth"))
    print("Phase1 finished")
    return model, log


def train_phase2_cls(
    model,
    source_loader,
    val_loader=None,
    device=None,
    epochs=40,
    lr=1e-3,
    save_dir=None,
    aux_weight=0.3,
    subband_reweight_mode="soft",
    source_pos_weight=1.0,
    early_stop_patience=0,
    early_stop_min_delta=0.0,
    early_stop_metric="subset_acc",
):
    _ensure_save_dir(save_dir)
    device = device or next(model.parameters()).device
    _set_trainable(model, train_encoder=False, train_domain=False)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    log = []
    best_score = None
    best_state = None
    stale_epochs = 0

    print("\n" + "=" * 50)
    print("Phase2: source frequency detector training")
    print(
        f"source_pos_weight={source_pos_weight} | "
        f"early_stop_patience={early_stop_patience} | "
        f"early_stop_metric={early_stop_metric}",
        flush=True,
    )
    print("=" * 50)

    for epoch in range(epochs):
        avg_loss = _train_source_epoch(
            model,
            source_loader,
            optimizer,
            device,
            subband_reweight_mode=subband_reweight_mode,
            cls_pos_weight=source_pos_weight,
        )

        if val_loader is not None:
            val_stats = evaluate_multiview_classification(
                model,
                val_loader,
                device,
                subband_reweight_mode=subband_reweight_mode,
            )
            log.append([
                epoch + 1,
                avg_loss,
                val_stats["element_acc"],
                val_stats["subset_acc"],
            ])
            print(
                f"Phase2 | Epoch [{epoch + 1}/{epochs}] | "
                f"loss:{avg_loss:.4f} | {_format_val_stats(val_stats)}",
                flush=True,
            )
            if early_stop_patience > 0:
                score = _early_stop_score(val_stats, early_stop_metric)
                improved = best_score is None or score > best_score + float(early_stop_min_delta)
                if improved:
                    best_score = score
                    best_state = _copy_state_dict(model)
                    stale_epochs = 0
                    if save_dir is not None:
                        torch.save(model.state_dict(), os.path.join(save_dir, "phase2_cls_trained_best.pth"))
                else:
                    stale_epochs += 1
                    if stale_epochs >= int(early_stop_patience):
                        print(
                            f"Phase2 early stopped at epoch {epoch + 1}; "
                            f"best_{early_stop_metric}={best_score:.6f}",
                            flush=True,
                        )
                        break
        else:
            log.append([epoch + 1, avg_loss])
            print(f"Phase2 | Epoch [{epoch + 1}/{epochs}] | loss:{avg_loss:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    if save_dir is not None:
        torch.save(model.state_dict(), os.path.join(save_dir, "phase2_cls_trained.pth"))
    print("Phase2 finished")
    return model, log


def train_phase2_finetune(
    model,
    source_loader,
    val_loader=None,
    device=None,
    epochs=10,
    lr_encoder=1e-5,
    lr_classifier=1e-4,
    save_dir=None,
    unfreeze_last_blocks=1,
    low_snr_threshold=None,
    low_snr_weight=1.0,
    aux_weight=0.3,
    subband_reweight_mode="soft",
    source_pos_weight=1.0,
    early_stop_patience=0,
    early_stop_min_delta=0.0,
    early_stop_metric="subset_acc",
):
    _ensure_save_dir(save_dir)
    device = device or next(model.parameters()).device

    for param in model.parameters():
        param.requires_grad = False

    for encoder in _frequency_encoders(model):
        _unfreeze_last_encoder_blocks(encoder, unfreeze_last_blocks)

    for name, param in model.named_parameters():
        if "freq_classifier" in name or "fusion_projector" in name or "subband_reweight" in name:
            param.requires_grad = True

    encoder_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "freq_encoder" in name:
            encoder_params.append(param)
        else:
            classifier_params.append(param)

    optimizer = optim.Adam(
        [
            {"params": encoder_params, "lr": lr_encoder},
            {"params": classifier_params, "lr": lr_classifier},
        ]
    )
    log = []
    best_score = None
    best_state = None
    stale_epochs = 0

    print("\n" + "=" * 50)
    print("Phase2.5: source frequency detector finetuning")
    print(
        f"source_pos_weight={source_pos_weight} | "
        f"early_stop_patience={early_stop_patience} | "
        f"early_stop_metric={early_stop_metric}",
        flush=True,
    )
    print("=" * 50)

    for epoch in range(epochs):
        avg_loss = _train_source_epoch(
            model,
            source_loader,
            optimizer,
            device,
            subband_reweight_mode=subband_reweight_mode,
            low_snr_threshold=low_snr_threshold,
            low_snr_weight=low_snr_weight,
            cls_pos_weight=source_pos_weight,
        )

        if val_loader is not None:
            val_stats = evaluate_multiview_classification(
                model,
                val_loader,
                device,
                subband_reweight_mode=subband_reweight_mode,
            )
            log.append([
                epoch + 1,
                avg_loss,
                val_stats["element_acc"],
                val_stats["subset_acc"],
            ])
            print(
                f"Phase2.5 | Epoch [{epoch + 1}/{epochs}] | "
                f"loss:{avg_loss:.4f} | {_format_val_stats(val_stats)}",
                flush=True,
            )
            if early_stop_patience > 0:
                score = _early_stop_score(val_stats, early_stop_metric)
                improved = best_score is None or score > best_score + float(early_stop_min_delta)
                if improved:
                    best_score = score
                    best_state = _copy_state_dict(model)
                    stale_epochs = 0
                    if save_dir is not None:
                        torch.save(model.state_dict(), os.path.join(save_dir, "phase2_finetuned_best.pth"))
                else:
                    stale_epochs += 1
                    if stale_epochs >= int(early_stop_patience):
                        print(
                            f"Phase2.5 early stopped at epoch {epoch + 1}; "
                            f"best_{early_stop_metric}={best_score:.6f}",
                            flush=True,
                        )
                        break
        else:
            log.append([epoch + 1, avg_loss])
            print(f"Phase2.5 | Epoch [{epoch + 1}/{epochs}] | loss:{avg_loss:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    if save_dir is not None:
        torch.save(model.state_dict(), os.path.join(save_dir, "phase2_finetuned.pth"))
    print("Phase2.5 finished")
    return model, log


def train_phase3_da(
    model,
    source_loader,
    target_loader,
    source_val_loader=None,
    device=None,
    epochs=30,
    lr=1e-4,
    save_dir=None,
    init_domain_weight=0.005,
    max_domain_weight=0.05,
    detach_condition=True,
    subband_reweight_mode="soft",
    grl_warmup_epochs=0,
    grl_warmup_value=0.0,
    domain_warmup_epochs=0,
    domain_warmup_value=None,
    aux_weight=0.3,
    pseudo_weight=0.02,
    num_subbands=8,
    pseudo_active_threshold=0.9,
    pseudo_inactive_threshold=0.1,
    pseudo_min_confidence=0.6,
    pseudo_gate_mode="fixed",
    pseudo_quantile_rho=0.25,
    pseudo_quantile_update="frozen",
    pseudo_quantile_active_floor=0.5,
    pseudo_quantile_inactive_ceiling=0.05,
    prototype_weight=0.05,
    source_anchor_weight=0.1,
    train_shared_projector=False,
    use_occupancy_condition=True,
):
    """Occupancy-semantic adversarial adaptation.

    The source detector is frozen as an anchor. During phase3, only the target
    encoder is adapted by adversarial domain matching, conservative subband
    pseudo labels, subband active/inactive prototype alignment, and optional
    energy-based target calibration.
    """
    _ensure_save_dir(save_dir)
    device = device or next(model.parameters()).device

    source_model = copy.deepcopy(model).to(device)
    _freeze_all(source_model)

    _configure_target_encoder_adaptation(model, train_shared_projector=train_shared_projector)
    discriminator = OccupancyDomainDiscriminator(
        feature_dim=getattr(model, "shared_dim", 128),
        num_subbands=num_subbands,
        use_occupancy_condition=use_occupancy_condition,
    ).to(device)

    trainable_target_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(
        [
            {"params": trainable_target_params, "lr": lr},
            {"params": discriminator.parameters(), "lr": lr},
        ]
    )
    log = []

    print("\n" + "=" * 50)
    print("Phase3: occupancy-semantic adversarial adaptation")
    print(
        f"source_anchor_weight={source_anchor_weight} | "
        f"domain_weight={init_domain_weight}->{max_domain_weight} | "
        f"pseudo_weight={pseudo_weight} | prototype_weight={prototype_weight}",
        flush=True,
    )
    print(
        f"use_occupancy_condition={use_occupancy_condition} | "
        f"pseudo active/inactive={pseudo_active_threshold}/{pseudo_inactive_threshold} | "
        f"min_confidence={pseudo_min_confidence} | "
        f"gate_mode={pseudo_gate_mode} | quantile_rho={pseudo_quantile_rho} | "
        f"quantile_update={pseudo_quantile_update} | "
        f"active_floor={pseudo_quantile_active_floor} | inactive_ceiling={pseudo_quantile_inactive_ceiling} | "
        f"train_shared_projector={train_shared_projector}",
        flush=True,
    )
    print("=" * 50)

    frozen_threshold_table = None
    if pseudo_gate_mode == "snr_quantile" and pseudo_quantile_update == "frozen":
        frozen_threshold_table = _estimate_snr_quantile_thresholds(
            model,
            target_loader,
            device,
            rho=pseudo_quantile_rho,
            num_subbands=num_subbands,
            subband_reweight_mode=subband_reweight_mode,
            fallback_active=pseudo_active_threshold,
            fallback_inactive=pseudo_inactive_threshold,
            active_floor=pseudo_quantile_active_floor,
            inactive_ceiling=pseudo_quantile_inactive_ceiling,
        )
        if frozen_threshold_table:
            print(
                "Phase3 frozen adaptive gate | "
                f"{_format_threshold_table(frozen_threshold_table)}",
                flush=True,
            )
        else:
            print(
                "Phase3 frozen adaptive gate requested but target SNR was unavailable; "
                "falling back to fixed thresholds.",
                flush=True,
            )

    for epoch in range(epochs):
        threshold_table = frozen_threshold_table
        if pseudo_gate_mode == "snr_quantile" and pseudo_quantile_update == "epoch":
            threshold_table = _estimate_snr_quantile_thresholds(
                model,
                target_loader,
                device,
                rho=pseudo_quantile_rho,
                num_subbands=num_subbands,
                subband_reweight_mode=subband_reweight_mode,
                fallback_active=pseudo_active_threshold,
                fallback_inactive=pseudo_inactive_threshold,
                active_floor=pseudo_quantile_active_floor,
                inactive_ceiling=pseudo_quantile_inactive_ceiling,
            )
            if threshold_table:
                active_values = [item[0] for item in threshold_table.values()]
                inactive_values = [item[1] for item in threshold_table.values()]
                print(
                    "Phase3 adaptive gate | "
                    f"{_format_threshold_table(threshold_table)}",
                    flush=True,
                )
            else:
                print(
                    "Phase3 adaptive gate requested but target SNR was unavailable; "
                    "falling back to fixed thresholds.",
                    flush=True,
                )
        elif pseudo_gate_mode == "snr_quantile" and pseudo_quantile_update != "frozen":
            raise ValueError(f"Unsupported pseudo_quantile_update: {pseudo_quantile_update}")
        elif pseudo_gate_mode != "fixed" and pseudo_gate_mode != "snr_quantile":
            raise ValueError(f"Unsupported pseudo_gate_mode: {pseudo_gate_mode}")

        model.train()
        source_model.eval()
        discriminator.train()
        _set_frozen_detector_eval(model)

        grl_lambda = _grl_schedule(
            epoch,
            epochs,
            warmup_epochs=grl_warmup_epochs,
            warmup_value=grl_warmup_value,
        )
        domain_weight = _linear_schedule(
            epoch,
            epochs,
            init_domain_weight,
            max_domain_weight,
            warmup_epochs=domain_warmup_epochs,
            warmup_value=domain_warmup_value,
        )

        totals = {
            "loss": 0.0,
            "anchor": 0.0,
            "domain": 0.0,
            "pseudo": 0.0,
            "prototype": 0.0,
            "pseudo_cov": 0.0,
            "pseudo_active_cov": 0.0,
            "pseudo_inactive_cov": 0.0,
            "proto_active_pairs": 0.0,
            "proto_inactive_pairs": 0.0,
        }
        num_steps = 0

        for source_batch, target_batch in zip(source_loader, target_loader):
            x_s, labels_s = _unpack_source_batch(source_batch, device)
            x_t, snr_t = _unpack_target_batch_with_snr(target_batch, device)

            optimizer.zero_grad()

            with torch.no_grad():
                logits_s_anchor, aux_s_anchor, _, _ = _forward_model(
                    source_model,
                    x_s,
                    subband_reweight_mode=subband_reweight_mode,
                    return_features=True,
                )

            logits_s_target, _, _, _ = _forward_model(
                model,
                x_s,
                subband_reweight_mode=subband_reweight_mode,
                return_features=True,
            )
            logits_t, aux_t, _, _ = _forward_model(
                model,
                x_t,
                subband_reweight_mode=subband_reweight_mode,
                return_features=True,
            )

            source_condition = None
            target_condition = None
            if use_occupancy_condition:
                source_condition = torch.sigmoid(logits_s_anchor).detach()
                target_condition = torch.sigmoid(logits_t).detach() if detach_condition else torch.sigmoid(logits_t)

            domain_s = discriminator(
                aux_s_anchor["shared_feature"].detach(),
                source_condition,
                grl_lambda=None,
            )
            domain_t = discriminator(
                aux_t["shared_feature"],
                target_condition,
                grl_lambda=grl_lambda,
            )
            label_s = torch.zeros(domain_s.size(0), dtype=torch.long, device=device)
            label_t = torch.ones(domain_t.size(0), dtype=torch.long, device=device)
            loss_domain = 0.5 * (
                nn.functional.cross_entropy(domain_s, label_s)
                + nn.functional.cross_entropy(domain_t, label_t)
            )

            active_threshold, inactive_threshold = _make_threshold_matrix(
                snr_t,
                threshold_table,
                pseudo_active_threshold,
                pseudo_inactive_threshold,
                num_subbands,
                logits_t.device,
            )
            loss_anchor = classification_loss(logits_s_target, labels_s)
            loss_pseudo, pseudo_stats = conservative_subband_pseudo_loss(
                logits_t,
                active_threshold=active_threshold,
                inactive_threshold=inactive_threshold,
                min_confidence=pseudo_min_confidence,
            )
            loss_prototype, proto_stats = subband_occupancy_prototype_loss(
                aux_s_anchor["subband_feature"],
                aux_t["subband_feature"],
                labels_s,
                logits_t,
                active_threshold=active_threshold,
                inactive_threshold=inactive_threshold,
                min_confidence=pseudo_min_confidence,
            )

            loss = (
                source_anchor_weight * loss_anchor
                + domain_weight * loss_domain
                + pseudo_weight * loss_pseudo
                + prototype_weight * loss_prototype
            )
            loss.backward()
            optimizer.step()

            totals["loss"] += loss.item()
            totals["anchor"] += loss_anchor.item()
            totals["domain"] += loss_domain.item()
            totals["pseudo"] += loss_pseudo.item()
            totals["prototype"] += loss_prototype.item()
            totals["pseudo_cov"] += float(pseudo_stats["pseudo_coverage"].item())
            totals["pseudo_active_cov"] += float(pseudo_stats["pseudo_active_coverage"].item())
            totals["pseudo_inactive_cov"] += float(pseudo_stats["pseudo_inactive_coverage"].item())
            totals["proto_active_pairs"] += float(proto_stats["prototype_active_pairs"].item())
            totals["proto_inactive_pairs"] += float(proto_stats["prototype_inactive_pairs"].item())
            num_steps += 1

        denom = max(num_steps, 1)
        row = [
            epoch + 1,
            totals["loss"] / denom,
            totals["anchor"] / denom,
            totals["domain"] / denom,
            totals["pseudo"] / denom,
            totals["prototype"] / denom,
            totals["pseudo_cov"] / denom,
            totals["pseudo_active_cov"] / denom,
            totals["pseudo_inactive_cov"] / denom,
            totals["proto_active_pairs"] / denom,
            totals["proto_inactive_pairs"] / denom,
            grl_lambda,
            domain_weight,
        ]

        if source_val_loader is not None:
            val_stats = evaluate_multiview_classification(
                model,
                source_val_loader,
                device,
                subband_reweight_mode=subband_reweight_mode,
            )
            row.extend([val_stats["element_acc"], val_stats["subset_acc"]])

        log.append(row)
        msg = (
            f"Phase3 | Epoch [{epoch + 1}/{epochs}] | "
            f"grl:{grl_lambda:.3f} domain_w:{domain_weight:.4f} | "
            f"total:{row[1]:.4f} anchor:{row[2]:.4f} domain:{row[3]:.4f} "
            f"pseudo:{row[4]:.4f} proto:{row[5]:.4f} | "
            f"cov:{row[6]:.3f} a_cov:{row[7]:.3f} i_cov:{row[8]:.3f} "
            f"proto_pairs:{row[9]:.1f}/{row[10]:.1f}"
        )
        if source_val_loader is not None:
            msg += f" | val_elem:{row[-2]:.4f} val_subset:{row[-1]:.4f}"
        print(msg, flush=True)

    if save_dir is not None:
        torch.save(model.state_dict(), os.path.join(save_dir, "final_occupancy_semantic_adapt_model.pth"))
        torch.save(discriminator.state_dict(), os.path.join(save_dir, "occupancy_domain_discriminator.pth"))
    print("Phase3 finished")
    return model, log





