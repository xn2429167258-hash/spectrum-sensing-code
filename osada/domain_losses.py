import torch
import torch.nn as nn


class CompareGRL(torch.autograd.Function):
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


class DANNDomainDiscriminator(nn.Module):
    def __init__(self, feature_dim=128, num_domains=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_domains),
        )

    def forward(self, feature, grl_lambda=1.0):
        return self.net(CompareGRL.apply(feature, grl_lambda))


class CDANDomainDiscriminator(nn.Module):
    def __init__(self, feature_dim=128, num_subbands=8, num_domains=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_subbands = num_subbands
        self.net = nn.Sequential(
            nn.Linear(feature_dim * num_subbands, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_domains),
        )

    def forward(self, feature, condition_prob, grl_lambda=1.0, detach_condition=True):
        condition = condition_prob.detach() if detach_condition else condition_prob
        cdan_input = condition.unsqueeze(2) * feature.unsqueeze(1)
        cdan_input = cdan_input.flatten(1)
        return self.net(CompareGRL.apply(cdan_input, grl_lambda))


def coral_loss(source_feature, target_feature):
    source_feature = source_feature.float()
    target_feature = target_feature.float()

    ns = source_feature.size(0)
    nt = target_feature.size(0)
    if ns <= 1 or nt <= 1:
        return source_feature.new_tensor(0.0)

    source_centered = source_feature - source_feature.mean(dim=0, keepdim=True)
    target_centered = target_feature - target_feature.mean(dim=0, keepdim=True)
    cov_s = source_centered.t().matmul(source_centered) / float(ns - 1)
    cov_t = target_centered.t().matmul(target_centered) / float(nt - 1)
    d = source_feature.size(1)
    return (cov_s - cov_t).pow(2).sum() / (4.0 * d * d)


def _gaussian_kernel_matrix(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    total = torch.cat([x, y], dim=0)
    total0 = total.unsqueeze(0)
    total1 = total.unsqueeze(1)
    l2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma is not None:
        bandwidth = float(fix_sigma)
    else:
        n_samples = total.size(0)
        denom = max(n_samples * n_samples - n_samples, 1)
        bandwidth = l2_distance.detach().sum() / float(denom)
        bandwidth = bandwidth.clamp_min(1e-6)

    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    kernels = []
    for i in range(kernel_num):
        kernels.append(torch.exp(-l2_distance / (bandwidth * (kernel_mul ** i)).clamp_min(1e-6)))
    return sum(kernels)


def mmd_loss(source_feature, target_feature, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    source_feature = source_feature.float()
    target_feature = target_feature.float()
    ns = source_feature.size(0)
    nt = target_feature.size(0)
    if ns <= 0 or nt <= 0:
        return source_feature.new_tensor(0.0)

    kernels = _gaussian_kernel_matrix(
        source_feature,
        target_feature,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )
    xx = kernels[:ns, :ns]
    yy = kernels[ns:, ns:]
    xy = kernels[:ns, ns:]
    yx = kernels[ns:, :ns]
    return xx.mean() + yy.mean() - xy.mean() - yx.mean()


def build_compare_discriminator(method, feature_dim=128, num_subbands=8, num_domains=2):
    if method == "dann":
        return DANNDomainDiscriminator(feature_dim=feature_dim, num_domains=num_domains)
    if method == "cdan":
        return CDANDomainDiscriminator(
            feature_dim=feature_dim,
            num_subbands=num_subbands,
            num_domains=num_domains,
        )
    if method in {"coral", "mmd"}:
        return None
    raise ValueError(f"Unsupported compare DA method: {method}")


