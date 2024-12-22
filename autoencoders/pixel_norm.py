import torch 
from torch import nn 

# ------------------------------------------------------------------------------
# class PixelNorm(nn.Module)
#     |   
# init ( dim = 1, eps=1e-8)
#     |
# initialize the input 
#     |
# fn forward(x)
#     |
# return (PixelNorm)(x) = x / (sqrt((mean)(x^2) + epsilon))

# ------------------------------------------------------------------------------


class PixelNorm(nn.Module):

    def __init__(self, dim = 1, eps = 1e-8):
        self.dim = dim 
        self.eps = eps 


    def forward(self, x):
        pixelnorm = x / torch.sqrt(torch.mean(input=x**2, dim=self.dim, keepdim=True) + self.eps)
        return pixelnorm


