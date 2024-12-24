import torch
from torch.nn import nn 
from torch.nn import functional as F


class SpatialNorm(nn.Module):
    """
    Spatially conditioned normalization as defined in https://arxiv.org/abs/2209.09002.

    Args:
        f_channels (`int`):
            The number of channels for input to group normalization layer, and output of the spatial norm layer.
        zq_channels (`int`):
            The number of channels for the quantized vector as described in the paper.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=32, eps=1e-6, affine=True)
        self.conv_y = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        f_size = f.shape[-2:]
        zq = F.interpolate(zq, size=f_size, mode="nearest")
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f
    
class LoRAAttnProcessor:
    r"""
    Processor for implementing attention with LoRA.
    """

    def __init__(self):
        pass


class LoRAAttnProcessor2_0:
    r"""
    Processor for implementing attention with LoRA (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        pass


class LoRAXFormersAttnProcessor:
    r"""
    Processor for implementing attention with LoRA using xFormers.
    """

    def __init__(self):
        pass


class LoRAAttnAddedKVProcessor:
    r"""
    Processor for implementing attention with LoRA with extra learnable key and value matrices for the text encoder.
    """

    def __init__(self):
        pass