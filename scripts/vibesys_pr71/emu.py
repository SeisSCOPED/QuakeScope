"""oneDNN fused conv(+ReLU) numerics, for torch builds without oneDNN."""

import torch
import torch.nn.functional as F


class EmuMKLDNN:
    """torch.ops.mkldnn's fused conv(+ReLU) with oneDNN numerics in the input dtype (fp32, bf16, fp16):
    fp32 accumulate, fp32 bias, ReLU, output rounded to the input dtype. Runs on any CPU."""

    @staticmethod
    def _convolution_pointwise(x, w, b, padding, stride, dilation, groups, attr, scalars, algo):
        y = F.conv2d(x.float(), w.float(), b.float(), stride=stride, padding=padding,
                     dilation=dilation, groups=groups)
        if attr == "relu":
            y = F.relu(y)
        return y.to(x.dtype).contiguous(memory_format=torch.channels_last)

    @staticmethod
    def _convolution_transpose_pointwise(x, w, b, padding, output_padding, stride, dilation,
                                         groups, attr, scalars, algo):
        y = F.conv_transpose2d(x.float(), w.float(), b.float(), stride=stride, padding=padding,
                               output_padding=output_padding, groups=groups, dilation=dilation)
        if attr == "relu":
            y = F.relu(y)
        return y.to(x.dtype).contiguous(memory_format=torch.channels_last)
