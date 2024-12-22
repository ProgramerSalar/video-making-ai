import math 
from typing import Tuple, Union

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from einops import rearrange


# ------------------------------------------------------------------------------------------------------------

## class DualConv3d 
# init (in_channels, out_channels, kernel_size, stride: Union[int,  Tuple] = 1, 
#       padding = 0, dilation = 1, groups=1, bias=True)
#     |
# in_channels, out_channels 
#     |
# Ensure kernel_size, stride, padding and dilation are tuple of length 3 
#     |
# groups, bias 
#     |
# define the intermidiate channels for the first convolutions 
#     |
# Define parameters for the first convolutions (intermediate_channels, in_channels // groups, 1, kernel_size[1][2])
#     |
# stride1[1, 1, 2], padding1[0, 1, 2], dilations1[1, 1, 2] 
#     |
# condition of bias
#     |
# Defne parameters for the second convolutions (out_channels, intermediate_channel // groups, kernel_size[0], 1, 1)
#     |   
# stride2[0, 1, 1], padding2[0, 0, 0], dilation2[0, 1, 1] 
#     |
# condition of bias
#     |
# initialize weight and biases using self.reset_parameters()
# _________________________________________________________________________________________________________________________________________

# fn reset parameters 
#     |
# 2 x kaiming_uniform(a = math.sqrt(5))
#     |
# if bias 
#     |
# 2 x calculate function_input_1 using _calculate_fan_in_and_fan_out
#     |
# 2 x define the range for the uniform distribution object is bound
#     |   
# 2 x uniform the distribution 
#     |

# ____________________________________________________________________________________________________________________________________________


# function forward (x, use_conv3d=False, skip_time_conv=False)
#     |
# if use conv3d 
#     |
# fun forward_with_3d(x, skip_time_conv)
#     |
# else 
#     |
# fun forward_with_2d(x, skip_time_conv)

# _______________________________________________________________________________________________________________________________________________

# fun forward_with_3d 
#     |
# first convolution
#     |
# return x if skip_time_conv is True 
#     |
# second convolution
#     |
# return x 

# __________________________________________________________________________________________________________________________________________________

# fn forward_with_2d
#     |
# batch_size, channels, depth, height, width = shape of x 
#     |
#     ## first 2d conv
# rearrange b c d h w -> (b d) c h w
#     |
# squeeze the depth dim out of weight, squeeze(2) of weight1
#     |
# select stride, padding, dilation for the 2d conv
#     |
# conv2d 
#     |
# _, _, h, w = x.shape 
#     |
# rearrange (b=b) the original shape with condition of skip_time_conv is true and then return x 
#     |
#     ## second 1d conv 
# rearange second conv ((b d) c h w -> (b h w) c d) b=b
#     |
# Reshape weight2 to match the expected dim for conv1d
#     |
# Use only the relevent dim for stride, padding and dilation for the 1d conv1d
#     |
# conv1d 
#     |
# rearrange (b h w) c d -> b c d h w and b, h, w 
#     |
# return x 

# ___________________________________________________________________________________________________________________________________________________

# @property
# fn weight 
#     |
# return self.weight2

# _____________________________________________________________________________________________________________________________________________________

# fn test_dual_conv3d_consistency
#     |
# in_channels=3, out_channels=5, kernel_size = 3, stride = 2, padding = 1
#     |
# create an instance of the DualConv3d class 
#     |
# Example input tensor (1, 3, 10, 10, 10)
#     |
# Perform forward passes with both 3d and 2d settings 
#     |
# Assert that the output from both methods are sufficiently close

# _____________________________________________________________________________________________________________________________________

# ------------------------------------------------------------------------------------------------------------


class DualConv3d:

    def __init__(self,
                in_channels,
                 out_channels,
                 kernel_size, 
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding:Union[int, Tuple[int, int, int]] = 0,
                 dilation:Union[int, Tuple[int, int, int]] = 1,
                 groups=1,
                 bias=True):
        


        self.in_channels = in_channels
        self.out_channels = out_channels


        # Ensure kernel_size, stride, padding and dilation are tuple of length 3 
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)

        if kernel_size == (1, 1, 1):
            raise ValueError("kernel size are greater than 1. Use make_linear_nd instead.")
        

        if isinstance(stride, int):
            self.stride = (stride, stride, stride)


        if isinstance(padding, int):
            self.padding = (padding, padding, padding)

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation, dilation)


        self.groups = groups
        self.bias = bias

        # Define the intermidiate channels for the first conv 
        intermidiate_channels = (
           out_channels if in_channels < out_channels else in_channels
        )


        # define parameters for the first conv 
        self.weight1 = nn.Parameter(
            torch.Tensro(
                intermidiate_channels,
                in_channels // groups,
                1, 
                kernel_size[1],
                kernel_size[2]
            )
        )



        self.stride1 = (1, stride[1], stride[2])
        self.padding1 = (0, padding[1], padding[2])
        self.dilation1 = (1, dilation[1], dilation[2])


        if bias:
            self.bias1 = nn.Parameter(torch.Tensor(intermidiate_channels))

        else:
            self.register_parameters("bias1", None)


        # define parameters for the second conv 
        self.weight2 = nn.Parameter(
            torch.Tensor(
                out_channels,
                intermidiate_channels // groups,
                kernel_size[0],
                1, 1
            )
        )


        self.stride2 = (stride[0], 1, 1)
        self.padding2 = (padding[0], 0, 0)
        self.dilation2 = (dilation[0], 1, 1)

        if bias:
            self.bias2 = nn.Parameter(torch.Tensor(out_channels))

        else:
            self.register_parameters("bias2", None)


        # init weight and bias
        self.register_parameters()


    def register_parameters(self):

        nn.init.kaiming_uniform_(tensor=self.weight1, 
                                 a=math.sqrt(5))
        
        nn.init.kaiming_uniform_(tensor=self.weight2,
                                 a=math.sqrt(5))
        

        if self.bias:

            fn_in1, _ = nn.init._calculate_fan_in_and_fan_out(tensor=self.weight1)
            bound = 1 / math.sqrt(fn_in1)
            torch.nn.init.uniform_(tensor=self.bias1, a=-bound, b=bound)

            fn_in2, _ = nn.init._calculate_fan_in_and_fan_out(tensor=self.weight2)
            bound2 = 1 / math.sqrt(fn_in2)
            torch.nn.init.uniform_(tensor=self.bias2, a=-bound2, b=bound2)





    def forward(self, x, use_conv3d=False, skip_time_conv=False):

        if use_conv3d:
            return self.forward_with_3d(x, skip_time_conv=skip_time_conv)
        

        else:
            return self.forward_with_2d(x, skip_time_conv=skip_time_conv)
        


    def forward_with_3d(self, x, skip_time_conv):

        # first conv 
        x = torch.nn.Conv3d(
            in_channels=x,
            out_channels=self.weight1,
            kernel_size=self.bias1,
            stride=self.stride1,
            padding=self.padding1,
            dilation=self.dilation1,
            groups=1,
            bias=True,
            padding_mode="zeros"
        )


        if skip_time_conv:
            return x 
        
        # second conv 
        x = torch.nn.Conv3d(
            in_channels=x,
            out_channels=self.weight2,
            kernel_size=self.bias2,
            stride=self.stride2,
            padding=self.padding2,
            dilation=self.dilation2,
            groups=1,
            bias=True,
            padding_mode="zeros"
        )


        return x 
    


    def forward_with_2d(self, x, skip_time_conv):

        b, c, d, h, w = x.shape 

        # first 2d conv 
        x = rearrange(tensor=x,
                      str="b c d h w -> (b d) c h w")
        

        # squeeze the depth dim out of weight 
        weight1 = self.weight1.squeeze(2)

        # select stride, padding, dilation for the 2d conv 
        stride1 = (self.stride1[1], self.stride1[2])
        padding1 = (self.padding1[1], self.padding1[2])
        dilation1 = (self.dilation1[1], self.dilation1[2])

        x = torch.nn.Conv2d(in_channels=x,
                            out_channels=weight1,
                            kernel_size=self.bias1,
                            stride=stride1,
                            padding=padding1,
                            dilation=dilation1)
        

        _, _, h, w = x.shape 

        if skip_time_conv:
            x = rearrange(tensor=x, 
                          str="(b d) c h w -> b c d h w")
            
            return x 
        

        # second 1d conv 
        x = rearrange(tensor=x, 
                      str="(b d) c h w -> (b h w) c d", b = b)
        
        # reshape weight2 to match the expected dim for conv1d 
        weight2 = self.weight2.unsqueeze(-1).unsqueeze(-1)


        # use only the relevent dim for stride, padding and dilation for the 1d conv1d 
        stride2 = self.stride2[0]
        padding2 = self.padding2[0]
        dilation2 = self.dilation2[0]


        x = torch.nn.Conv1d(
            in_channels=x,
            out_channels=weight2,
            kernel_size=self.bias2,
            stride=stride2,
            padding=padding2,
            dilation=dilation2
        )


        return x 
    


    @property
    def weight(self):
        return self.weight2
    


    def test_dual_conv3d_consistency(self):

        in_channels = 3
        out_channels = 5
        kernel_size = (3, 3, 3) 
        stride = (2, 2, 2)
        padding = (1, 1, 1)

        # create an instance of the DualConv3d Class 
        dual_conv3d = DualConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ) 


        # Example input tensor 
        input_tensor = (1, 3, 10, 10, 10)

        # perform forward passes with both 3d and 2d settings 
        dualconv_3d = dual_conv3d(input_tensor, use_conv3d=True)
        dualconv_2d = dual_conv3d(input_tensor, use_conv3d=False)


        assert torch.allclose(
            input=dualconv_3d,
            other=dualconv_2d,
            rtol=1e-8
        ),  "outputs are not consistent between 3d and 2d conv."

        



















        



            












        


        

        
        












