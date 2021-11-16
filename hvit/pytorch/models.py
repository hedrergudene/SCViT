import torch
import numpy as np
from HViT_classification.hvit.pytorch.functions import *
from typing import List

# HViT
class HViT(torch.nn.Module):
    def __init__(self,
                 img_size:int=32,
                 patch_size:List[int]=[2,4,8],
                 num_channels:int=3,
                 projection_dim:int=192,
                 depth:int=4,
                 num_heads:int=8,
                 mlp_head_units:List[int]=[128],
                 num_classes:int=100,
                 hidden_dim_factor:float=2.,
                 attn_drop:float=.2,
                 proj_drop:float=.2,
                 linear_drop:float=.2,
                 upsampling_type:str="conv",
                 original_attn:bool=True,
                 dtype:torch.dtype=torch.float,
                 ):
        super(HViT, self).__init__()
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.projection_dim = projection_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_head_units = [self.projection_dim] + mlp_head_units + [num_classes]
        self.hidden_dim_factor = hidden_dim_factor
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.linear_drop = linear_drop
        self.upsampling_type = upsampling_type
        self.original_attn = original_attn
        self.dtype = dtype
        ## Parameters computations
        self.num_patches = [(self.img_size//ps)**2 for ps in self.patch_size]

        # Layers
        self.PE = PatchEncoder(self.img_size, self.patch_size[0], self.num_channels, self.projection_dim, self.dtype)
        self.Encoders = torch.nn.ModuleList()
        for i in range(len(self.num_patches)):
            # Block of Transformer Encoders
            for _ in range(self.depth):
                self.Encoders.append(
                    TransformerEncoder(self.img_size,
                                       self.patch_size[i],
                                       self.num_channels,
                                       self.projection_dim,
                                       self.hidden_dim_factor,
                                       self.num_heads,
                                       self.attn_drop,
                                       self.proj_drop,
                                       self.linear_drop,
                                       self.original_attn,
                                       )
                )
            # Resampling
            if (i+1)<len(self.num_patches):
                self.Encoders.append(
                    Upsampling(
                           self.img_size,
                           self.patch_size[i:i+2],
                           self.num_channels,
                           self.projection_dim,
                           self.upsampling_type,
                       )

                )

        self.LN = torch.nn.LayerNorm(normalized_shape=(self.num_patches[-1], self.projection_dim))
        self.MLP = torch.nn.ModuleList()
        for j in range(len(self.mlp_head_units)-1):
            self.MLP.append(torch.nn.Linear(self.mlp_head_units[j], self.mlp_head_units[j+1]))
            if (j+2)<len(self.mlp_head_units):
                self.MLP.append(torch.nn.Dropout(self.linear_drop))

    def forward(self, X:torch.Tensor):
        encoded = self.PE(X)
        for layer in self.Encoders:
            encoded = layer(encoded)
        encoded = self.LN(encoded)
        encoded = torch.mean(encoded, dim = 1)
        for linear in self.MLP:
            encoded = linear(encoded)
        return encoded