import torch
from torch import nn, Tensor
from zeta import SSM


class CobraBlock(nn.Module):
    def __init__(self, dim: int, dt_rank: int, dim_inner : int,d_state :int):
        super().__init__()


        ## Projecion
        self.prof = nn.Linear(dim, dim)
        
        ## Convolution
        self.conv = nn.Conv1d(dim, dim, kernel_size = 3, padding = 1, groups = dim)

        ##Activation
        self.silu = nn.SiLU()

        ## init SSM

        self.ssm = SSM(dim, dt_rank, dim_inner, d_state)

    
    def forward(self, x: Tensor):
        ## Create 2 pathways

        skip = x

        ## Split up the patches

        x_one = self.proj(x)
        x_two = self.prof(x)

        ## Apply the convlution
        x_one = self.conv(x_one)

        ## Apply the activation
        x_one = self.silu(x_one)

        ## Apply the SSM
        x_one = self.ssm(x_one)

        #Apply the activation
        x_two = self.silu(x_two)

        ## Matmul

        out = x_one * x_two.T

        ## Add the skip connection

        out = out + skip

        return self.proj(out)
    

x = torch.randn(1, 64, 256)

block = CobraBlock(64, 8, 128, 256)

out = block(x)
print(out)
