import torch
import torch.nn.functional as F


def _as_threshold(value, reference):
    if torch.is_tensor(value):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(float(value), device=reference.device, dtype=reference.dtype)


def classification_loss(pred, target, pos_weight=None, sample_weight=None):
    target = target.to(device=pred.device, dtype=pred.dtype)
    if sample_weight is None:
        return F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)

    loss = F.binary_cross_entropy_with_logits(
        pred,
        target,
        pos_weight=pos_weight,
        reduction="none",
    )
    sample_loss = loss.mean(dim=1)
    sample_weight = sample_weight.to(device=pred.device, dtype=pred.dtype)
    return (sample_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)


def domain_loss(pred, label):
    label = label.to(device=pred.device, dtype=torch.long)
    return F.cross_entropy(pred, label)


def source_target_domain_loss(domain_out_s, domain_out_t):
    label_s = torch.zeros(domain_out_s.size(0), dtype=torch.long, device=domain_out_s.device)
    label_t = torch.ones(domain_out_t.size(0), dtype=torch.long, device=domain_out_t.device)
    return 0.5 * (domain_loss(domain_out_s, label_s) + domain_loss(domain_out_t, label_t))


def weighted_domain_adaptation_loss(
    global_domain_out_s,
    global_domain_out_t,
    global_weight=1.0,
):
    if global_domain_out_s is not None:
        device = global_domain_out_s.device
    elif global_domain_out_t is not None:
        device = global_domain_out_t.device
    else:
        device = torch.device("cpu")

    if global_domain_out_s is None or global_domain_out_t is None:
        loss_global = torch.tensor(0.0, device=device)
    else:
        loss_global = source_target_domain_loss(global_domain_out_s, global_domain_out_t)

    weighted = global_weight * loss_global
    items = {
        "global": loss_global.detach(),
        "weighted": weighted.detach(),
    }
    return weighted, items


def pseudo_label_loss(logits, confidence=None, threshold=0.8):
    pseudo_label = (torch.sigmoid(logits) >= 0.5).to(dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, pseudo_label, reduction="none")
    sample_loss = loss.mean(dim=1)

    if confidence is None:
        mask = torch.ones_like(sample_loss)
    else:
        confidence = confidence.to(device=sample_loss.device, dtype=sample_loss.dtype)
        mask = (confidence >= threshold).to(dtype=sample_loss.dtype) * confidence

    return (sample_loss * mask).sum() / mask.sum().clamp_min(1e-6)


def conservative_subband_pseudo_loss(
    logits,
    active_threshold=0.9,
    inactive_threshold=0.1,
    min_confidence=0.6,
):
    """Subband-wise conservative pseudo-label loss for unlabeled target data.

    Only high-confidence active and inactive subbands are used. Ambiguous
    subbands are ignored, which prevents low-confidence active bins from being
    self-trained as inactive.
    """
    probs = torch.sigmoid(logits)
    confidence = torch.maximum(probs, 1.0 - probs)
    active_threshold = _as_threshold(active_threshold, probs)
    inactive_threshold = _as_threshold(inactive_threshold, probs)
    active_mask = probs >= active_threshold
    inactive_mask = probs <= inactive_threshold
    reliable_mask = confidence >= float(min_confidence)
    mask = (active_mask | inactive_mask) & reliable_mask

    pseudo = torch.zeros_like(logits)
    pseudo[active_mask] = 1.0

    element_loss = F.binary_cross_entropy_with_logits(logits, pseudo, reduction="none")
    weights = mask.to(dtype=logits.dtype) * confidence.detach()
    loss = (element_loss * weights).sum() / weights.sum().clamp_min(1e-6)

    stats = {
        "pseudo_coverage": weights.detach().gt(0).to(dtype=logits.dtype).mean(),
        "pseudo_active_coverage": active_mask.detach().to(dtype=logits.dtype).mean(),
        "pseudo_inactive_coverage": inactive_mask.detach().to(dtype=logits.dtype).mean(),
        "pseudo_mean_confidence": confidence.detach()[mask].mean() if mask.any() else logits.new_tensor(0.0),
    }
    return loss, stats


def subband_occupancy_prototype_loss(
    source_subband_feature,
    target_subband_feature,
    source_labels,
    target_logits,
    active_threshold=0.9,
    inactive_threshold=0.1,
    min_confidence=0.6,
    normalize=True,
):
    """Align target subband features to source active/inactive prototypes.

    Features are shaped [B, C, K], labels/logits are [B, K]. For each subband,
    source active and inactive prototypes are computed from labeled source
    samples, while target prototypes use only conservative pseudo-labels.
    """
    if source_subband_feature.dim() != 3 or target_subband_feature.dim() != 3:
        raise ValueError("Expected subband features shaped [batch, channels, num_subbands].")

    src_feat = source_subband_feature
    tgt_feat = target_subband_feature
    if normalize:
        src_feat = F.normalize(src_feat, dim=1)
        tgt_feat = F.normalize(tgt_feat, dim=1)

    source_labels = source_labels.to(device=src_feat.device, dtype=src_feat.dtype)
    target_probs = torch.sigmoid(target_logits)
    target_conf = torch.maximum(target_probs, 1.0 - target_probs)
    active_threshold = _as_threshold(active_threshold, target_probs)
    inactive_threshold = _as_threshold(inactive_threshold, target_probs)
    if active_threshold.dim() == 0:
        active_threshold = active_threshold.expand_as(target_probs)
    if inactive_threshold.dim() == 0:
        inactive_threshold = inactive_threshold.expand_as(target_probs)

    losses = []
    active_pairs = 0
    inactive_pairs = 0
    num_subbands = source_labels.size(1)

    for band_idx in range(num_subbands):
        src_active = source_labels[:, band_idx] >= 0.5
        src_inactive = ~src_active
        tgt_active = (
            (target_probs[:, band_idx] >= active_threshold[:, band_idx])
            & (target_conf[:, band_idx] >= float(min_confidence))
        )
        tgt_inactive = (
            (target_probs[:, band_idx] <= inactive_threshold[:, band_idx])
            & (target_conf[:, band_idx] >= float(min_confidence))
        )

        if src_active.any() and tgt_active.any():
            src_proto = src_feat[src_active, :, band_idx].mean(dim=0).detach()
            tgt_proto = tgt_feat[tgt_active, :, band_idx].mean(dim=0)
            losses.append(F.mse_loss(tgt_proto, src_proto))
            active_pairs += 1

        if src_inactive.any() and tgt_inactive.any():
            src_proto = src_feat[src_inactive, :, band_idx].mean(dim=0).detach()
            tgt_proto = tgt_feat[tgt_inactive, :, band_idx].mean(dim=0)
            losses.append(F.mse_loss(tgt_proto, src_proto))
            inactive_pairs += 1

    if losses:
        loss = torch.stack(losses).mean()
    else:
        loss = target_logits.new_tensor(0.0)

    stats = {
        "prototype_active_pairs": target_logits.new_tensor(float(active_pairs)),
        "prototype_inactive_pairs": target_logits.new_tensor(float(inactive_pairs)),
    }
    return loss, stats


def _target_subband_log_power(iq, num_subbands, eps=1e-6):
    complex_x = torch.complex(iq[:, 0], iq[:, 1])
    spec = torch.fft.fft(complex_x, dim=-1)
    spec = torch.fft.fftshift(spec, dim=-1)
    power = torch.log1p(torch.abs(spec) ** 2)
    batch_size, fft_bins = power.shape
    if fft_bins % int(num_subbands) != 0:
        raise ValueError(f"FFT bins ({fft_bins}) must be divisible by num_subbands ({num_subbands}).")
    tile_bins = fft_bins // int(num_subbands)
    tile_power = power.view(batch_size, int(num_subbands), tile_bins).mean(dim=-1)
    centered = tile_power - tile_power.median(dim=1, keepdim=True).values
    scale = centered.abs().median(dim=1, keepdim=True).values.clamp_min(eps)
    return centered / scale


def np_cfar_target_loss(
    logits_t,
    x_t,
    num_subbands=8,
    decision_threshold=0.01,
    idle_quantile=0.25,
    active_quantile=0.75,
    idle_weight=1.0,
    active_weight=1.0,
):
    """NP/CFAR-inspired unsupervised target calibration.

    Target labels are not used. Candidate idle/active subbands are estimated
    from relative target-domain spectral energy. The loss suppresses false
    activation on idle candidates and encourages detection on strong active
    candidates.
    """
    probs = torch.sigmoid(logits_t)
    with torch.no_grad():
        rel_energy = _target_subband_log_power(x_t, num_subbands=num_subbands)
        idle_cut = torch.quantile(rel_energy, float(idle_quantile), dim=1, keepdim=True)
        active_cut = torch.quantile(rel_energy, float(active_quantile), dim=1, keepdim=True)
        idle_mask = (rel_energy <= idle_cut).to(dtype=logits_t.dtype)
        active_mask = (rel_energy >= active_cut).to(dtype=logits_t.dtype)
        active_mask = active_mask * (1.0 - idle_mask)

    idle_count = idle_mask.sum().clamp_min(1.0)
    active_count = active_mask.sum().clamp_min(1.0)

    idle_excess = F.relu(probs - float(decision_threshold))
    idle_loss = ((idle_excess ** 2) * idle_mask).sum() / idle_count

    active_targets = torch.ones_like(logits_t)
    active_loss_raw = F.binary_cross_entropy_with_logits(logits_t, active_targets, reduction="none")
    active_loss = (active_loss_raw * active_mask).sum() / active_count

    total = float(idle_weight) * idle_loss + float(active_weight) * active_loss
    stats = {
        "np_cfar_idle_loss": idle_loss.detach(),
        "np_cfar_active_loss": active_loss.detach(),
        "np_cfar_idle_coverage": idle_mask.detach().mean(),
        "np_cfar_active_coverage": active_mask.detach().mean(),
        "np_cfar_idle_score": (probs.detach() * idle_mask).sum() / idle_count,
        "np_cfar_active_score": (probs.detach() * active_mask).sum() / active_count,
    }
    return total, stats


def domain_adaptation_loss(
    cls_out_s,
    labels_s,
    domain_out_s,
    domain_out_t,
    cls_weight=1.0,
    domain_weight=0.1,
):
    loss_cls = classification_loss(cls_out_s, labels_s)
    if domain_out_s is None or domain_out_t is None:
        loss_domain_global = torch.zeros_like(loss_cls)
    else:
        loss_domain_global = source_target_domain_loss(domain_out_s, domain_out_t)

    weighted_domain = domain_weight * loss_domain_global
    total = cls_weight * loss_cls + weighted_domain
    items = {
        "cls": loss_cls.detach(),
        "domain": weighted_domain.detach(),
        "domain_global": loss_domain_global.detach(),
        "total": total.detach(),
    }
    return total, items


