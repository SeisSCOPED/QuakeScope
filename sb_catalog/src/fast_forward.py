"""
A fused inference graph for SeisBench's PhaseNet forward pass.

The bf16-autocast forward of `sbm.PhaseNet` spends about 85% of `classify` in
the network itself, and most of that at the two full-resolution levels (16
channels x 3001 samples), where each conv is memory-bound: oneDNN reorders
every activation from PyTorch's plain NCW layout into its blocked layout and
back, and BatchNorm, ReLU and the skip concatenation each make another full
pass over the tensor.

`PhaseNetGraph` is the same network rebuilt for inference only:

- every eval-mode BatchNorm is folded into the conv that feeds it (the model
  never trains here, so the running statistics are constants);
- the 1-D convs run as (1, k) 2-D convs on channels-last tensors, the layout
  oneDNN's bf16 kernels use natively, so no activation is reordered;
- weights are stored in the compute dtype once, instead of autocast casting
  them on every batch; the folded bias stays fp32 and absorbs the mean shift
  that rounding the weights to bf16 would add (data-free bias correction from
  BatchNorm's statistics);
- each conv and its ReLU run as one oneDNN primitive;
- with deconv="polyphase", each stride-4 transposed conv runs as one stride-1
  conv with 4x the output channels, one per output phase (oneDNN's bf16
  deconvolution primitive is several times slower per MAC than its conv); the
  phases are interleaved by a channels-last view, so the skip concatenation
  that follows stays the only copy. The weights are the transposed conv's own
  (rounded, bias-corrected) weights, rearranged, so the two arms differ only
  in accumulation order.

The numerics are PhaseNet's to bf16 rounding; the probabilities are returned
as fp32 NCW, exactly the shape and dtype `annotate_batch_post` expects.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _fold(conv: torch.nn.Module, bn: torch.nn.Module | None
          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (weight, bias) of `conv` with the eval-mode `bn` after it folded in."""
    weight = conv.weight.detach().double()
    bias = (conv.bias.detach().double() if conv.bias is not None
            else torch.zeros(weight.shape[1 if isinstance(conv, torch.nn.ConvTranspose1d) else 0],
                             dtype=torch.float64))
    if bn is not None:
        scale = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
        if isinstance(conv, torch.nn.ConvTranspose1d):
            weight = weight * scale[None, :, None]  # (in, out, k)
        else:
            weight = weight * scale[:, None, None]  # (out, in, k)
        bias = (bias - bn.running_mean.double()) * scale + bn.bias.detach().double()
    return weight.float(), bias.float()


def _relu_mean(bn: torch.nn.Module) -> torch.Tensor:
    """E[ReLU(y)] per channel for y ~ N(bn.bias, bn.weight**2), BN's own output model."""
    mu = bn.bias.detach().double()
    sigma = bn.weight.detach().double().abs().clamp_min(1e-12)
    z = mu / sigma
    normal = torch.distributions.Normal(0.0, 1.0)
    return (mu * normal.cdf(z) + sigma * torch.exp(normal.log_prob(z))).float()


def _bias_correction(weight: torch.Tensor, rounded: torch.Tensor, input_mean: torch.Tensor,
                     kind: str, stride: int) -> torch.Tensor:
    """Bias that cancels the mean output shift from rounding `weight` to `rounded`.

    Data-free bias correction (Nagel et al., 2019, "Data-Free Quantization"): with
    E[x_i] per input channel, rounding error dW shifts every output of channel c
    by sum_{i,k} dW[c, i, k] E[x_i]; the fp32 bias absorbs the opposite. A
    transposed conv with stride s gives each output k/s taps on average.
    """
    error = (rounded - weight).double()
    if kind == "conv":  # (out, in, k)
        return -(error.sum(-1) @ input_mean.double()).float()
    return -((input_mean.double() @ error.sum(-1)) / stride).float()  # (in, out, k)


#: oneDNN's fused conv(+ReLU) ops, as torch.compile's CPU backend emits them. They
#: take a bf16 input and weight with an fp32 bias; F.conv2d requires the bias in the
#: input dtype, and a bf16 bias is a systematic per-channel error once BatchNorm's
#: shift is folded into it. Without them the graph falls back to F.conv2d + ReLU.
_MKLDNN = getattr(torch.ops, "mkldnn", None)
FUSED_OPS = all(hasattr(_MKLDNN, op) for op in
                ("_convolution_pointwise", "_convolution_transpose_pointwise"))

#: How the decoder's transposed convs run (QS_DECONV in pick_workload.py):
#: "transpose" is oneDNN's deconvolution; "polyphase" is the same layer as a
#: stride-1 conv over output phases plus a zero-copy interleave.
DECONV_IMPLS = ("transpose", "polyphase")


def _polyphase(weight4: torch.Tensor, bias: torch.Tensor, stride: int
               ) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Rearrange a transposed conv into a stride-1 conv over its output phases.

    A transposed conv with weight W (in, out, 1, k) and stride s gives
    y[s*m + r] = b + sum_j W[..., r + s*j] x[m - j]. Phase r is a stride-1 conv
    with taps W[r + s*(taps-1)], ..., W[r] (zero past k) and padding taps - 1,
    so stacking the s phases gives one (s*out, in, 1, taps) conv whose output
    position m, channel r*out + c is y[s*m + r] of channel c. Returns the
    weight (channels-last), the bias repeated per phase and the padding.
    """
    cin, cout, _, k = weight4.shape
    taps = -(-k // stride)
    weight = weight4.new_zeros(stride, cout, cin, 1, taps)
    for r in range(stride):
        for j in range(taps):
            if r + stride * j < k:
                weight[r, :, :, 0, taps - 1 - j] = weight4[:, :, 0, r + stride * j].T
    weight = weight.reshape(stride * cout, cin, 1, taps)
    return weight.contiguous(memory_format=torch.channels_last), bias.repeat(stride), taps - 1


class PhaseNetGraph(torch.nn.Module):
    """Inference-only PhaseNet: BN folded, (1, k) convs, channels-last."""

    def __init__(self, model: torch.nn.Module, dtype: torch.dtype = torch.bfloat16,
                 fused: bool = FUSED_OPS, deconv: str = "transpose") -> None:
        super().__init__()
        if deconv not in DECONV_IMPLS:
            raise ValueError(f"deconv impl {deconv!r} is not one of {DECONV_IMPLS}")
        self.dtype = dtype
        self.fused = fused
        self.deconv = deconv
        self._params = torch.nn.ParameterList()

        def add(conv, bn, kind, stride=1, padding=0, relu=True, input_mean=None):
            weight, bias = _fold(conv, bn)
            weight4 = weight.unsqueeze(2).to(dtype)  # (o, i, 1, k) or (i, o, 1, k)
            if input_mean is not None and dtype != torch.float32:
                bias = bias + _bias_correction(weight, weight4.squeeze(2).float(), input_mean,
                                               kind, stride)
            if kind == "convt" and deconv == "polyphase":
                weight4, bias, padding = _polyphase(weight4, bias, stride)
                kind, stride = "conv", 1
            elif kind == "conv":
                weight4 = weight4.contiguous(memory_format=torch.channels_last)
            w = torch.nn.Parameter(weight4, requires_grad=False)
            b = torch.nn.Parameter(bias if fused else bias.to(dtype), requires_grad=False)
            self._params.append(w)
            self._params.append(b)
            return (kind, w, b, stride, padding, relu)

        # The normalized waveform has zero mean; every later layer reads ReLU(BN(.)).
        self.inc = add(model.inc, model.in_bn, "conv", 1, 3)
        mean = _relu_mean(model.in_bn)
        self.down = []
        skip_means = []
        for conv_same, bn1, conv_down, bn2 in model.down_branch:
            same = add(conv_same, bn1, "conv", 1, 3, input_mean=mean)
            mean = _relu_mean(bn1)
            down = None
            if conv_down is not None:
                skip_means.append(mean)
                down = add(conv_down, bn2, "conv", 4, conv_down.padding[0], input_mean=mean)
                mean = _relu_mean(bn2)
            self.down.append((same, down))
        self.up = []
        #: Per up level, the transposed conv's (stride, kernel) when it runs as phases.
        self.up_phases = []
        for (conv_up, bn1, conv_same, bn2), skip_mean in zip(model.up_branch, skip_means[::-1]):
            up = add(conv_up, bn1, "convt", 4, 0, input_mean=mean)
            self.up_phases.append((4, conv_up.kernel_size[0]) if deconv == "polyphase" else None)
            same = add(conv_same, bn2, "conv", 1, 3,
                       input_mean=torch.cat([skip_mean, _relu_mean(bn1)]))
            mean = _relu_mean(bn2)
            self.up.append((up, same))
        self.out = add(model.out, None, "conv", 1, 0, relu=False, input_mean=mean)

    def _apply(self, layer, x: torch.Tensor) -> torch.Tensor:
        kind, w, b, stride, padding, relu = layer
        if self.fused:
            attr = "relu" if relu else "none"
            if kind == "conv":
                return _MKLDNN._convolution_pointwise(
                    x, w, b, [0, padding], [1, stride], [1, 1], 1, attr, [], "")
            return _MKLDNN._convolution_transpose_pointwise(
                x, w, b, [0, 0], [0, 0], [1, stride], [1, 1], 1, attr, [], "")
        if kind == "conv":
            x = F.conv2d(x, w, b, stride=(1, stride), padding=(0, padding))
        else:
            x = F.conv_transpose2d(x, w, b, stride=(1, stride))
        return F.relu(x, inplace=True) if relu else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, 3, samples) float -> (batch, 3, samples) fp32 probabilities."""
        x = x.to(self.dtype).unsqueeze(2).contiguous(memory_format=torch.channels_last)
        x = self._apply(self.inc, x)
        skips = []
        for i, (same, down) in enumerate(self.down):
            x = self._apply(same, x)
            if down is not None:
                skips.append(x)
                if i == 1:
                    x = F.pad(x, (2, 3))
                elif i == 2:
                    x = F.pad(x, (1, 3))
                elif i == 3:
                    x = F.pad(x, (2, 3))
                x = self._apply(down, x)
        for (conv_up, conv_same), phases, skip in zip(self.up, self.up_phases, skips[::-1]):
            length = x.shape[-1]
            x = self._apply(conv_up, x)
            if phases is not None:
                # (B, s*o, 1, M) channels-last is (B, M, s, o) in memory: as (B, o, 1, s*M)
                # it is the transposed conv's output, padded to s*M samples.
                stride, kernel = phases
                n, c, _, m = x.shape
                x = x.permute(0, 2, 3, 1).view(n, 1, m * stride, c // stride).permute(0, 3, 1, 2)
                x = x[..., :(length - 1) * stride + kernel]
            x = x[..., 1:-2]
            offset = (x.shape[-1] - skip.shape[-1]) // 2
            x = torch.cat([skip, x[..., offset:offset + skip.shape[-1]]], dim=1)
            x = self._apply(conv_same, x)
        x = self._apply(self.out, x)
        return torch.softmax(x.float(), dim=1).squeeze(2).contiguous()
