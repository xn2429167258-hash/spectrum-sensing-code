import math
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from osada.losses import classification_loss
from osada.domain_losses import (
    build_compare_discriminator,
    coral_loss,
    mmd_loss,
)
from osada.source_training import evaluate_multiview_classification


def _ensure_save_dir(save_dir):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)


def _to_device(x, device):
    return x.to(device, non_blocking=True)


def _unpack_source_batch(batch, device):
    x, labels = batch[0], batch[1]
    return _to_device(x, device), _to_device(labels, device)


def _unpack_target_batch(batch, device):
    if isinstance(batch, (tuple, list)):
        x = batch[0]
    else:
        x = batch
    return _to_device(x, device)


def _grl_schedule(epoch, epochs, warmup_epochs=0, warmup_value=0.0):
    if epochs <= 1:
        return 1.0
    warmup_epochs = max(0, min(int(warmup_epochs), max(epochs - 1, 0)))
    if epoch < warmup_epochs:
        return float(warmup_value)
    p = float(epoch - warmup_epochs) / float(max(epochs - 1 - warmup_epochs, 1))
    return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def _linear_schedule(epoch, epochs, init_value, max_value, warmup_epochs=0, warmup_value=None):
    warmup_epochs = max(0, int(warmup_epochs))
    if epoch < warmup_epochs:
        return float(init_value if warmup_value is None else warmup_value)
    if epochs <= warmup_epochs + 1:
        return float(max_value)
    p = float(epoch - warmup_epochs) / float(max(epochs - 1 - warmup_epochs, 1))
    return float(init_value + (max_value - init_value) * p)


def _domain_loss(domain_out_s, domain_out_t):
    label_s = torch.zeros(domain_out_s.size(0), dtype=torch.long, device=domain_out_s.device)
    label_t = torch.ones(domain_out_t.size(0), dtype=torch.long, device=domain_out_t.device)
    return 0.5 * (F.cross_entropy(domain_out_s, label_s) + F.cross_entropy(domain_out_t, label_t))


def _forward_backbone(model, x, subband_reweight_mode="soft"):
    logits, aux, _ = model(
        x,
        grl_lambda=0.0,
        use_conditional_domain=False,
        subband_reweight_mode=subband_reweight_mode,
    )
    return logits, aux


def train_phase3_da_compare(
    model,
    source_loader,
    target_loader,
    source_val_loader=None,
    method="dann",
    device=None,
    epochs=30,
    lr=1e-4,
    save_dir=None,
    init_domain_weight=0.05,
    max_domain_weight=0.3,
    grl_warmup_epochs=0,
    grl_warmup_value=0.0,
    domain_warmup_epochs=0,
    domain_warmup_value=None,
    aux_weight=0.3,
    subband_reweight_mode="soft",
    detach_condition=True,
    feature_dim=128,
    num_subbands=8,
    num_domains=2,
):
    if method not in {"dann", "cdan", "coral", "mmd"}:
        raise ValueError(f"Unsupported compare DA method: {method}")

    _ensure_save_dir(save_dir)
    device = device or next(model.parameters()).device

    discriminator = build_compare_discriminator(
        method,
        feature_dim=feature_dim,
        num_subbands=num_subbands,
        num_domains=num_domains,
    )
    if discriminator is not None:
        discriminator = discriminator.to(device)

    params = list(model.parameters())
    if discriminator is not None:
        params += list(discriminator.parameters())
    optimizer = optim.Adam(params, lr=lr)
    log = []

    print("\n" + "=" * 50)
    print(f"Phase3 frequency-only compare DA | method={method}")
    print(
        f"subband_reweight_mode={subband_reweight_mode} | "
        f"grl_warmup_epochs={grl_warmup_epochs} | "
        f"domain_warmup_epochs={domain_warmup_epochs}",
        flush=True,
    )
    print("=" * 50)

    for epoch in range(epochs):
        model.train()
        if discriminator is not None:
            discriminator.train()

        total_loss = 0.0
        total_cls = 0.0
        total_domain = 0.0
        num_steps = 0

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

        for source_batch, target_batch in zip(source_loader, target_loader):
            x_s, labels_s = _unpack_source_batch(source_batch, device)
            x_t = _unpack_target_batch(target_batch, device)

            optimizer.zero_grad()

            logits_s, aux_s = _forward_backbone(
                model,
                x_s,
                subband_reweight_mode=subband_reweight_mode,
            )
            _, aux_t = _forward_backbone(
                model,
                x_t,
                subband_reweight_mode=subband_reweight_mode,
            )

            loss_cls = classification_loss(logits_s, labels_s)

            if method == "coral":
                loss_domain = coral_loss(aux_s["shared_feature"], aux_t["shared_feature"])
            elif method == "mmd":
                loss_domain = mmd_loss(aux_s["shared_feature"], aux_t["shared_feature"])
            elif method == "dann":
                domain_s = discriminator(aux_s["shared_feature"], grl_lambda=grl_lambda)
                domain_t = discriminator(aux_t["shared_feature"], grl_lambda=grl_lambda)
                loss_domain = _domain_loss(domain_s, domain_t)
            else:
                domain_s = discriminator(
                    aux_s["shared_feature"],
                    aux_s["condition"],
                    grl_lambda=grl_lambda,
                    detach_condition=detach_condition,
                )
                domain_t = discriminator(
                    aux_t["shared_feature"],
                    aux_t["condition"],
                    grl_lambda=grl_lambda,
                    detach_condition=detach_condition,
                )
                loss_domain = _domain_loss(domain_s, domain_t)

            loss = loss_cls + domain_weight * loss_domain
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls += loss_cls.item()
            total_domain += loss_domain.item()
            num_steps += 1

        denom = max(num_steps, 1)
        avg_loss = total_loss / denom
        avg_cls = total_cls / denom
        avg_domain = total_domain / denom

        row = [
            epoch + 1,
            avg_loss,
            avg_cls,
            avg_domain,
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
            row.extend([
                val_stats["element_acc"],
                val_stats["subset_acc"],
            ])
            print(
                f"Phase3 compare {method} | Epoch [{epoch + 1}/{epochs}] | "
                f"grl:{grl_lambda:.3f} domain_w:{domain_weight:.3f} | "
                f"total:{avg_loss:.4f} cls:{avg_cls:.4f} domain:{avg_domain:.4f} | "
                f"val_elem:{val_stats['element_acc']:.4f} "
                f"val_subset:{val_stats['subset_acc']:.4f}",
                flush=True,
            )
        else:
            print(
                f"Phase3 compare {method} | Epoch [{epoch + 1}/{epochs}] | "
                f"grl:{grl_lambda:.3f} domain_w:{domain_weight:.3f} | "
                f"total:{avg_loss:.4f} cls:{avg_cls:.4f} domain:{avg_domain:.4f}",
                flush=True,
            )
        log.append(row)

    if save_dir is not None:
        torch.save(model.state_dict(), os.path.join(save_dir, f"final_{method}_adapt_model.pth"))
        if discriminator is not None:
            torch.save(discriminator.state_dict(), os.path.join(save_dir, f"{method}_domain_discriminator.pth"))
    print(f"Phase3 compare DA finished | method={method}")
    return model, log


