from typing import Tuple, Union
import torch 
from torch import nn 


# ---------------------------------------------------------------------------------------------------------------------
# Causal convoluation 3d (nn.Module)
    # |
# init (in_channels, out_channels, kernel_size, stride: Union[int, Tuple[int]] = 1, 
# dilation: int = 1, groups = 1, **kwargs)
    # |
# in_channels
    # |
# out_channels 
    # |
# 3 x kernel_size 
    # |
# object time kernel size index 0 of kernel size 
    # |
# dilation (dilation, height, width)
    # |
# height padding kernel size 1 divisible of 2
    # |
# width padding  kernel size 2 divisible of 2
    # |
# padding (0, height, width)
    # |
# conv (in_channels, out_channels, kernel_size, stride, dilation, padding, padding_mode, groups)
# ______________________________________________________________________________________________________________________________________________________________


# forward(x, causal:bool=True)
    # |
# if causal 
    # |
# first frame padding and repeat time kernel size - 1 
    # |
# concatenate dim=2 
    # |
# else 
    # |
# first frame pading and  repeat time kernel size -1 // 2 
    # |
# last frame padding[-1:] and repeat same 
    # |
# concatenate (frst, last, x) dim=2 
    # |
# pass the x in conv 
    # |
# return x 

# _________________________________________________________________________________________________________________________________________________________________


# define property
# weight fn 
#     |
# return conv weight 


# ---------------------------------------------------------------------------------------------------------------------





class CausalConv3d(nn.Module):

    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride:Union[int, Tuple[int]] = 1,
                 dilation:int=1,
                 groups=1,
                 **kwargs):
        

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = (kernel_size, kernel_size, kernel_size)

        self.time_kernel_size = self.kernel_size[0]
        self.dilation = (dilation, 1, 1)
        self.pad_height = self.kernel_size[1] // 2 
        self.pad_width = self.kernel_size[2] // 2

        self.padding = (0, self.pad_height, self.pad_width)


        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            dilation=self.dilation,
            padding=self.padding,
            padding_mode="zeros",
            groups=groups
        )


    



    def forward(self, x, causal:bool=True):

        if causal:

            first_frame_padding = x[:, :, :1, :, :].repeat(1, 1, self.time_kernel_size - 1, 1, 1)
            x = torch.cat(tensors=(first_frame_padding, x), dim=2)


        else:

            first_frame_padding = x[:, :, :1, :, :].repeat(1, 1, (self.time_kernel_size - 1) // 2, 1, 1)
            last_frame_padding = x[:, :, -1:, :, :].repeat(1, 1, (self.time_kernel_size - 1) // 2, 1, 1)

            x = torch.cat(tensors=(first_frame_padding, last_frame_padding, x), dim=2)

        x = self.conv(x)
        
        return x 


    @property
    def weight(self):
        return self.conv.weight
    




            
        
        

        