import torch
import torch.nn as nn
import torch.nn.functional as F


# Adapted from https://github.com/AlarioAI/BitNet/blob/main/bitnet/bitlinear.py
class BitLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        num_bits: int = 8,
    ):
        super().__init__(in_features, out_features, bias)
        self.eps:float = 1e-5
        self.quantization_range: int = 2 ** (num_bits - 1) # Q_b in the paper
        self.norm: nn.Module = nn.LayerNorm(in_features)


    def ste_weights(self, weights_gamma: float) -> torch.Tensor:
        eps: float = 1e-8
        scaled_weights:torch.Tensor = self.weight / (weights_gamma + eps)
        binarized_input_no_grad: torch.Tensor = torch.clamp(torch.round(scaled_weights), min=-1, max=1)
        binarized_input_with_grad: torch.Tensor = (binarized_input_no_grad - self.weight).detach() + self.weight
        return binarized_input_with_grad


    def binarize_weights(self, weights_gamma: float) -> torch.Tensor:
        binarized_weights = self.ste_weights(weights_gamma)
        return binarized_weights


    def quantize_activations(self, _input:torch.Tensor, input_gamma: float) -> torch.Tensor:
        # Equation 4 BitNet paper
        quantized_input = torch.clamp(
                _input * self.quantization_range / input_gamma,
                -self.quantization_range + self.eps,
                self.quantization_range - self.eps,
            )
        return quantized_input


    def dequantize_activations(self, _input: torch.Tensor, input_gamma: float, beta: float) -> torch.Tensor:
        return _input * input_gamma * beta / self.quantization_range


    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        normalized_input: torch.Tensor = self.norm(_input)
        input_gamma: float = normalized_input.abs().max().item()
        weight_abs_mean: float = self.weight.abs().mean().item()

        binarized_weights = self.binarize_weights(weight_abs_mean)
        input_quant = self.quantize_activations(normalized_input, input_gamma)
        output = F.linear(input_quant, binarized_weights, self.bias)
        output = self.dequantize_activations(output, input_gamma, weight_abs_mean)

        return output