import torch
import numpy as np
from HViT_classification.hvit.pytorch.functions import *
from typing import List

# HViT
class HViT(torch.nn.Module):
    def __init__(self,
                 img_size:int=32,
                 patch_size:List[int]=[2,4,8,16],
                 num_channels:int=3,
                 projection_dim:int=192,
                 depth:int=6,
                 num_heads:int=4,
                 mlp_head_units:List[int]=[128],
                 num_classes:int=100,
                 hidden_dim_factor:float=2.,
                 attn_drop:float=.05,
                 proj_drop:float=.05,
                 linear_drop:float=.2,
                 upsampling_type:str="hybrid",
                 dtype:torch.dtype=torch.float,
                 verbose:bool=False,
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
        self.dtype = dtype
        self.verbose = verbose
        ## Parameters computations
        self.num_patches = [(self.img_size//ps)**2 for ps in self.patch_size]

        # Layers
        self.PE = PatchEncoder(self.img_size, self.patch_size[0], self.num_channels, self.projection_dim, self.dtype)
        self.Encoders = torch.nn.ModuleList()
        self.Resampling = torch.nn.ModuleList()
        for i in range(len(self.num_patches)):
            # Block of Transformer Encoder
            self.Encoders.append(
                TransformerEncoderBlock(self.img_size,
                                        self.patch_size[i],
                                        self.num_channels,
                                        self.depth,
                                        self.projection_dim,
                                        self.hidden_dim_factor,
                                        self.num_heads,
                                        self.attn_drop,
                                        self.proj_drop,
                                        self.linear_drop,
                                        )
            )
            # Resampling
            if (i+1)<len(self.num_patches):
                self.Resampling.append(
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
        if self.verbose: print("After PE",encoded.shape)
        for i in range(len(self.Encoders)):
            if self.verbose: print(f"Step", i)
            encoded = self.Encoders[i](encoded)
            if self.verbose: print("\tAfter Attn",encoded.shape)
            if (i+1)<len(self.Encoders):
                encoded = self.Resampling[i](encoded)
                if self.verbose: print("\tAfter resampling",encoded.shape)
        encoded = self.LN(encoded)
        encoded = torch.mean(encoded, dim = 1)
        if self.verbose: print(f"Finished encoding. Shape after averaging:",encoded.shape)
        for linear in self.MLP:
            encoded = linear(encoded)
        return encoded