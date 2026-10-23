"""PA-TSMixer and its single-scale parent TSMixer for direct XYZ trajectory forecasting.

Both models map a standardized history x of shape [B, 60, 3] to a forecast of shape [B, H, 3]. With
``last_residual=True`` (the default of PA-TSMixer) the linear head predicts the displacement from the last observed
position, which is added to it; the TSMixer baseline outputs positions directly. Parameter names match the
released checkpoints; the residual adds no parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(length, width, dropout):
    return nn.Sequential(nn.Linear(length, width), nn.ReLU(), nn.Linear(width, length), nn.Dropout(dropout))


class MultiScaleTemporalMixing(nn.Module):
    """Full-, half- and quarter-resolution temporal MLPs fused by learned scale weights and a 1x1 convolution.

    Branch widths follow the sequence length (d, d/2, d/4) unless ``uniform_width`` is given.
    ``learn_scale_weights=False`` fixes the scale weights to 1/3 (equal-weight control).
    """

    def __init__(self, seq_len=60, channels=3, d_model=128, dropout=0.1, uniform_width=None, learn_scale_weights=True):
        super().__init__()
        half, quarter = seq_len // 2, seq_len // 4
        widths = (uniform_width,) * 3 if uniform_width else (d_model, max(d_model // 2, 64), max(d_model // 4, 32))
        self.fine = _mlp(seq_len, widths[0], dropout)
        self.med = _mlp(half, widths[1], min(dropout * 1.5, 0.5))
        self.coarse = _mlp(quarter, widths[2], min(dropout * 2.0, 0.5))
        self.fuse = nn.Conv1d(3 * channels, channels, kernel_size=1)
        self.scale_w = nn.Parameter(torch.zeros(3), requires_grad=learn_scale_weights)

    def forward(self, x):                       # x: [B, D, L]
        length = x.shape[-1]
        fine = self.fine(x)
        med = F.interpolate(self.med(F.avg_pool1d(x, 2, 2)), size=length, mode="linear", align_corners=False)
        coarse = F.interpolate(self.coarse(F.avg_pool1d(x, 4, 4)), size=length, mode="linear", align_corners=False)
        w = F.softmax(self.scale_w, dim=0)
        blended = w[0] * fine + w[1] * med + w[2] * coarse
        return blended + self.fuse(torch.cat([fine, med, coarse], dim=1))


class MixerBlock(nn.Module):
    def __init__(self, temporal, channels, d_model, dropout):
        super().__init__()
        self.temporal = temporal
        self.channel = nn.Sequential(nn.Linear(channels, d_model), nn.ReLU(), nn.Linear(d_model, channels),
                                     nn.Dropout(dropout))

    def forward(self, x):                       # x: [B, L, D]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        return x + self.channel(x)


class _Mixer(nn.Module):
    def __init__(self, blocks, seq_len, pred_len, last_residual):
        super().__init__()
        self.model = nn.ModuleList(blocks)
        self.projection = nn.Linear(seq_len, pred_len)
        self.last_residual = last_residual

    def forward(self, x):                       # x: [B, 60, 3] standardized
        last = x[:, -1:]
        for block in self.model:
            x = block(x)
        out = self.projection(x.transpose(1, 2)).transpose(1, 2)
        return out + last if self.last_residual else out


class PATSMixer(_Mixer):
    """PA-TSMixer (44,672 parameters for H=30 with the default settings)."""

    def __init__(self, pred_len, seq_len=60, channels=3, d_model=128, layers=2, dropout=0.1,
                 uniform_width=None, learn_scale_weights=True, last_residual=True):
        blocks = [MixerBlock(MultiScaleTemporalMixing(seq_len, channels, d_model, dropout, uniform_width,
                                                      learn_scale_weights), channels, d_model, dropout)
                  for _ in range(layers)]
        super().__init__(blocks, seq_len, pred_len, last_residual)


class TSMixer(_Mixer):
    """Single-scale TSMixer parent (34,724 parameters for H=30; d_model=167 gives the matched-size control)."""

    def __init__(self, pred_len, seq_len=60, channels=3, d_model=128, layers=2, dropout=0.1, last_residual=False):
        blocks = [MixerBlock(_mlp(seq_len, d_model, dropout), channels, d_model, dropout) for _ in range(layers)]
        super().__init__(blocks, seq_len, pred_len, last_residual)


def build(name, pred_len, **kwargs):
    return {"PA-TSMixer": PATSMixer, "TSMixer": TSMixer}[name](pred_len, **kwargs)
