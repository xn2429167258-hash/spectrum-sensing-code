import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Model3_NoNegTransfer implements a frequency-only detector with subband tiling.

The input is still the original IQ tensor. The model computes a full-band FFT
log-power spectrum internally, splits the 1024-bin spectrum into 8 non-
overlapping subband tiles, and runs the same frequency encoder on every tile.
This keeps the external API compatible with the previous FreqOnly model while
removing the full-band shortcut that made domain adaptation easy to over-align.
"""


def _make_norm_1d(num_channels, norm_type="bn", num_groups=8):
    norm_type = str(norm_type).lower()
    if norm_type == "bn":
        return nn.BatchNorm1d(num_channels)
    if norm_type == "in":
        return nn.InstanceNorm1d(num_channels, affine=True)
    if norm_type == "gn":
        groups = min(int(num_groups), int(num_channels))
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    if norm_type in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.pool(x).squeeze(-1)
        scale = self.fc(scale).unsqueeze(-1)
        return x * scale


class MultiScaleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(3, 9, 15), dilation=1):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=(kernel_size // 2) * dilation,
                    dilation=dilation,
                    bias=False,
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.fuse = nn.Conv1d(out_channels * len(kernel_sizes), out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.fuse(x)


class ResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        norm_type="bn",
        norm_groups=8,
        use_se=False,
        multiscale=False,
    ):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        if multiscale:
            self.conv1 = MultiScaleConv1D(in_channels, out_channels, dilation=dilation)
        else:
            self.conv1 = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            )
        self.bn1 = _make_norm_1d(out_channels, norm_type, num_groups=norm_groups)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = _make_norm_1d(out_channels, norm_type, num_groups=norm_groups)
        self.se = SEBlock1D(out_channels) if use_se else nn.Identity()

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                _make_norm_1d(out_channels, norm_type, num_groups=norm_groups),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        return F.relu(x + residual)


class FreqSerialEncoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        latent_channels=128,
        norm_type="bn",
        norm_groups=8,
        dilations=(1, 1, 1),
        use_se=False,
        multiscale=False,
    ):
        super().__init__()
        if len(dilations) != 3:
            raise ValueError(f"Expected three dilation values, got {dilations}")
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3, bias=False),
            _make_norm_1d(64, norm_type),
            nn.ReLU(),
            ResidualBlock1D(
                64,
                96,
                kernel_size=5,
                dilation=dilations[0],
                norm_type=norm_type,
                norm_groups=norm_groups,
                use_se=use_se,
                multiscale=multiscale,
            ),
            ResidualBlock1D(
                96,
                latent_channels,
                kernel_size=11,
                dilation=dilations[1],
                norm_type=norm_type,
                norm_groups=norm_groups,
                use_se=use_se,
                multiscale=multiscale,
            ),
            ResidualBlock1D(
                latent_channels,
                latent_channels,
                kernel_size=5,
                dilation=dilations[2],
                norm_type=norm_type,
                norm_groups=norm_groups,
                use_se=use_se,
                multiscale=multiscale,
            ),
        )

    def forward(self, x, return_pooled=False):
        feat = self.net(x)
        pooled = F.adaptive_avg_pool1d(feat, 1).flatten(1)
        if return_pooled:
            return feat, pooled
        return feat


class SubbandFreqHead(nn.Module):
    def __init__(self, in_channels, num_subbands):
        super().__init__()
        self.num_subbands = num_subbands
        self.subband_pool = nn.AdaptiveAvgPool1d(num_subbands)
        self.subband_refine = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=max(1, in_channels // 16),
                bias=False,
            ),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
        )
        self.band_classifier = nn.Conv1d(in_channels, 1, kernel_size=1)

    def forward(self, feat):
        subband_feat = self.subband_pool(feat)
        subband_feat = self.subband_refine(subband_feat)
        logits = self.band_classifier(subband_feat).squeeze(1)
        pooled = subband_feat.mean(dim=-1)
        return logits, pooled, subband_feat


class TileFreqHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.tile_refine = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=max(1, in_channels // 16),
                bias=False,
            ),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
        )
        self.tile_classifier = nn.Linear(in_channels, 1)

    def forward(self, tile_feat):
        tile_feat = self.tile_refine(tile_feat)
        pooled = F.adaptive_avg_pool1d(tile_feat, 1).flatten(1)
        logits = self.tile_classifier(pooled).squeeze(-1)
        return logits, pooled


class WidebandFreqOnlyCDA(nn.Module):
    def __init__(
        self,
        num_subbands=8,
        num_domains=2,
        latent_channels=128,
        shared_dim=128,
        subband_reweight_mode="soft",
        fft_norm_mode="log_power",
        encoder_norm="bn",
        encoder_gn_groups=8,
        encoder_dilations=(1, 1, 1),
        encoder_use_se=False,
        encoder_multiscale=False,
        **unused_kwargs,
    ):
        super().__init__()
        self.num_subbands = num_subbands
        self.num_domains = num_domains
        self.latent_channels = latent_channels
        self.shared_dim = shared_dim
        self.default_subband_reweight_mode = subband_reweight_mode
        self.fft_norm_mode = fft_norm_mode
        self.encoder_norm = encoder_norm
        self.encoder_gn_groups = int(encoder_gn_groups)
        self.encoder_dilations = tuple(int(v) for v in encoder_dilations)
        self.encoder_use_se = bool(encoder_use_se)
        self.encoder_multiscale = bool(encoder_multiscale)

        self.freq_encoder = FreqSerialEncoder(
            1,
            latent_channels=latent_channels,
            norm_type=encoder_norm,
            norm_groups=self.encoder_gn_groups,
            dilations=self.encoder_dilations,
            use_se=self.encoder_use_se,
            multiscale=self.encoder_multiscale,
        )
        self.freq_classifier = TileFreqHead(latent_channels)

        self.subband_reweight = nn.Sequential(
            nn.Linear(num_subbands * 2, num_subbands),
            nn.ReLU(),
            nn.Linear(num_subbands, num_subbands),
        )

        self.shared_projector = nn.Sequential(
            nn.Linear(latent_channels, 256),
            nn.ReLU(),
            nn.Linear(256, shared_dim),
            nn.ReLU(),
        )

    @staticmethod
    def spectral_transform(x, norm_mode="log_power", eps=1e-6):
        if norm_mode not in {"log_power", "zscore", "center"}:
            raise ValueError(f"Unsupported FFT normalization mode: {norm_mode}")
        complex_x = torch.complex(x[:, 0], x[:, 1])
        spec = torch.fft.fft(complex_x, dim=-1)
        spec = torch.fft.fftshift(spec, dim=-1)
        power = torch.log1p(torch.abs(spec) ** 2)
        if norm_mode == "zscore":
            power = (power - power.mean(dim=-1, keepdim=True)) / (power.std(dim=-1, keepdim=True) + eps)
        elif norm_mode == "center":
            power = power - power.mean(dim=-1, keepdim=True)
        return power.unsqueeze(1)

    def split_frequency_tiles(self, freq_x):
        batch_size, channels, fft_bins = freq_x.shape
        if channels != 1:
            raise ValueError(f"Expected single-channel FFT log-power, got {channels} channels.")
        if fft_bins % self.num_subbands != 0:
            raise ValueError(
                f"FFT bins ({fft_bins}) must be divisible by num_subbands ({self.num_subbands})."
            )
        tile_bins = fft_bins // self.num_subbands
        tiles = freq_x.view(batch_size, channels, self.num_subbands, tile_bins)
        tiles = tiles.permute(0, 2, 1, 3).contiguous()
        return tiles.view(batch_size * self.num_subbands, channels, tile_bins), tile_bins

    def encode_frequency(self, x, subband_reweight_mode=None):
        if subband_reweight_mode is None:
            subband_reweight_mode = self.default_subband_reweight_mode
        if subband_reweight_mode not in {"none", "soft"}:
            raise ValueError(f"Unsupported subband_reweight_mode: {subband_reweight_mode}")

        batch_size = x.size(0)
        freq_x = self.spectral_transform(x, norm_mode=self.fft_norm_mode)
        freq_tiles, tile_bins = self.split_frequency_tiles(freq_x)

        tile_feat = self.freq_encoder(freq_tiles)
        tile_logits, tile_pooled = self.freq_classifier(tile_feat)

        logits = tile_logits.view(batch_size, self.num_subbands)
        tile_pooled = tile_pooled.view(batch_size, self.num_subbands, self.latent_channels)
        freq_subband_map = tile_pooled.transpose(1, 2).contiguous()

        probs = torch.sigmoid(logits)
        certainty = torch.abs(probs - 0.5) * 2.0
        evidence = torch.cat([probs, certainty], dim=1)

        if subband_reweight_mode == "soft":
            subband_weights = torch.sigmoid(self.subband_reweight(evidence))
            pooled = torch.sum(freq_subband_map * subband_weights.unsqueeze(1), dim=-1)
            pooled = pooled / subband_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        else:
            subband_weights = torch.full(
                (logits.size(0), self.num_subbands),
                1.0 / float(self.num_subbands),
                device=logits.device,
                dtype=logits.dtype,
            )
            pooled = freq_subband_map.mean(dim=-1)

        return logits, pooled, freq_subband_map, probs, subband_weights

    def forward(
        self,
        x,
        subband_reweight_mode=None,
        return_features=False,
        **unused_da_kwargs,
    ):
        detect_out, freq_pooled, freq_subband_map, freq_prob, subband_weights = self.encode_frequency(
            x,
            subband_reweight_mode=subband_reweight_mode,
        )
        condition = torch.sigmoid(detect_out)
        shared_feat = self.shared_projector(freq_pooled)

        aux = {
            "logits": detect_out,
            "freq_logits": detect_out,
            "freq_prob": freq_prob,
            "subband_weights": subband_weights,
            "condition": condition,
            "shared_feature": shared_feat,
            "subband_feature": freq_subband_map,
            "freq_subband_map": freq_subband_map,
            "freq_feature": freq_pooled,
        }

        if return_features:
            return detect_out, aux, None, {
                "freq_feature": freq_pooled,
                "freq_subband_map": freq_subband_map,
                **aux,
            }
        return detect_out, aux, None


BroadBandDomainAdaptNet = WidebandFreqOnlyCDA


