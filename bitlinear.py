import torch.nn as nn
import torch.nn.functional as F
from model_zoo import RMSNorm


# Adapted from https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
def activation_quant(x):
    """ Per-token quantization to 8 bits. No grouping is needed for quantization.
    Args:
    x: an activation tensor with shape [n, d]
    Returns:
    y: a quantized activation tensor with shape [n, d]
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w):
    """ Per-tensor quantization to 1.58 bits. No grouping is needed for quantization.
    Args:
    w: a weight tensor with shape [d, k]
    Returns:
    u: a quantized weight with shape [d, k]
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u


class BitLinear(nn.Linear):
    """
    This is only for training, and kernel optimization is needed for efficiency.
    """
    def __init__(self, in_features, outfeatures):
        super().__init__(in_features, outfeatures, bias=False)
        self.rms_norm = RMSNorm(hidden_size=in_features)
    
    def forward(self, x):
        """
        Args:
        x: an input tensor with shape [n, d]
        Returns:
        y: an output tensor with shape [n, d]
        """
        w = self.weight # a weight tensor with shape [d, k]
        x_norm = self.rms_norm(x)
        # A trick for implementing Straight-Through-Estimator (STE) using detach()
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y
