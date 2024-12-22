from typing import Tuple, Union
import torch

from autoencoders.causal_conv3d import CausalConv3d
from autoencoders.daual_conv3d import DualConv3d

# ------------------------------------------------------------------------------------
# fn make_conv_nd (dim:Union, in_channels, out_channels,
#                   kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, causal=False)
#     |
# dims = 2 
#     |
# dim = 3 
#     |
# return Dualconv3d if dims = (2, 1) 
#     |
# else valueError 
# ______________________________________________________________________________________________________

# fn make_linear_nd (dims, in_channels, out_channels, bias)
#     |
# return conv2d if dims = 2 
#     |
# return conv3d if dims = 3 or dims = (2, 1)
#     |
# else valueError 
# __________________________________________________________________________________________
# ------------------------------------------------------------------------------------




def make_conv_nd(dims: Union[int, Tuple[int, int, int]],
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 causal=False
                 ):
    

    if dims == 2:
        return torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=bias

        )
    

    elif dims == 3:
        if causal:
            return CausalConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias
            )

        else:
            torch.nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias
            )



    elif dims == (2, 1):
        return DualConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
    


    else:
        raise ValueError("{dims} are not supported.")




def make_linear_nd(dims:int,
                   in_channels:int,
                   out_channels: int,
                   bias=True):
    

    if dims == 2:
        return torch.nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               bias=bias
                               )
    

    elif dims == 3 or dims == (2, 1):
        return torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias
        )
    

    else:
        raise ValueError("{dims} does not supported.")



