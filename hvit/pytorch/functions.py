## Upsampling
import torch
import numpy as np
from typing import List

# Functions
## Patch
def patch(X:torch.Tensor,
          patch_size:int,
          ):
    if len(X.size())==5:
        X = torch.squeeze(X, dim=1)
    h, w = X.shape[-2], X.shape[-1]
    assert h%patch_size==0, f"Patch size must divide images height"
    assert w%patch_size==0, f"Patch size must divide images width"
    patches_desc = X.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patch_list = torch.flatten(patches_desc, 2, 3).permute(0,2,1,3,4)
    return patch_list

## Unflatten
def unflatten(flattened, num_channels):
        # Alberto: Added to reconstruct from bs, n, projection_dim -> bs, n, c, h, w
        bs, n, p = flattened.size()
        unflattened = torch.reshape(flattened, (bs, n, num_channels, int(np.sqrt(p//num_channels)), int(np.sqrt(p//num_channels))))
        return unflattened

# Layers
## DoubleConv
class DoubleConv(torch.nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, num_channels:int, groups:int):
        super(DoubleConv, self).__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, num_channels, kernel_size=3, padding='same', groups = groups),
            torch.nn.BatchNorm2d(num_channels),
            torch.nn.GELU(),
            torch.nn.Conv2d(num_channels, num_channels, kernel_size=3, padding='same', groups = groups),
            torch.nn.BatchNorm2d(num_channels),
            torch.nn.GELU(),
        )

    def forward(self, x):
        return self.double_conv(x)

## Patch Encoder
class PatchEncoder(torch.nn.Module):
    def __init__(self,
                 img_size:int,
                 patch_size:int,
                 num_channels:int,
                 projection_dim:int=768,
                 dtype:torch.dtype=torch.float32,
                 device="cuda:0",
                 ):
        super(PatchEncoder, self).__init__()
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.dtype = dtype
        self.projection_dim = projection_dim
        self.num_patches = (self.img_size//self.patch_size)**2
        self.positions = torch.arange(start = 0,
                         end = self.num_patches,
                         step = 1,
                         ).to(device)

        # Layers
        self.linear = torch.nn.Linear(self.num_channels*self.patch_size**2, self.projection_dim, dtype=self.dtype) if projection_dim is not None else torch.nn.Identity(dtype=self.dtype)
        self.position_embedding = torch.nn.Embedding(num_embeddings=self.num_patches,
                                                     embedding_dim = self.projection_dim,
                                                     )

    def forward(self, X):
        patches = patch(X, self.patch_size)
        flat_patches = torch.flatten(patches, -3, -1)
        encoded = self.linear(flat_patches) + self.position_embedding(self.positions)
        return encoded

## FeedForward
class FeedForward(torch.nn.Module):
    def __init__(self,
                 projection_dim:int,
                 hidden_dim_factor:float,
                 dropout:float,
                 ):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(projection_dim, int(hidden_dim_factor*projection_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(int(hidden_dim_factor*projection_dim), projection_dim),
            torch.nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


## TransformerEncoder
class TransformerEncoderBlock(torch.nn.Module):
    def __init__(self,
                 img_size:int,
                 patch_size:int,
                 num_channels:int,
                 depth:int=6,
                 projection_dim:int=None,
                 hidden_dim_factor:float=2.,
                 num_heads:int=4,
                 attn_drop:float=.05,
                 proj_drop:float=.05,
                 linear_drop:float=.2,
                 ):
        super().__init__()
        ## Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (self.img_size//self.patch_size)**2
        self.num_channels = num_channels
        self.depth = depth
        self.projection_dim = projection_dim if projection_dim is not None else self.num_channels*self.patch_size**2
        self.hidden_dim_factor = hidden_dim_factor
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.linear_drop = linear_drop
        ## Layers
        self.Attn = torch.nn.ModuleList()
        for _ in range(self.depth):
            self.Attn.append(torch.nn.MultiheadAttention(projection_dim, self.num_heads, self.attn_drop, batch_first = True))
        self.LN1 = torch.nn.ModuleList()
        for _ in range(self.depth):
            self.LN1.append(torch.nn.LayerNorm(normalized_shape = (self.num_patches, self.projection_dim)))
        self.LN2 = torch.nn.ModuleList()
        for _ in range(self.depth):
            self.LN2.append(torch.nn.LayerNorm(normalized_shape = (self.num_patches, self.projection_dim)))
        self.FeedForward = torch.nn.ModuleList()
        for __ in range(self.depth):
            self.FeedForward.append(
                FeedForward(projection_dim = self.projection_dim,
                            hidden_dim_factor = self.hidden_dim_factor,
                            dropout = self.linear_drop,
                            )
            )

    def forward(self, encoded_patches):
        for i in range(self.depth):
            encoded_patch_attn, _ = self.Attn[i](encoded_patches, encoded_patches, encoded_patches)
            encoded_patches = encoded_patch_attn + encoded_patches
            encoded_patches = self.LN1[i](encoded_patches)
            encoded_patches = self.FeedForward[i](encoded_patches) + encoded_patches
            encoded_patches = self.LN2[i](encoded_patches)
        return encoded_patches

## Upsampling
class Upsampling(torch.nn.Module):
    def __init__(self,
                 img_size:int,
                 patch_size:List[int],
                 num_channels:int,
                 projection_dim:int=768,
                 upsampling_type:str='hybrid',
                 device="cuda:0",
                 ):
        super(Upsampling, self).__init__()
        # Validation
        assert upsampling_type in ['hybrid', 'hybrid_channel'], f"Upsampling type must either be 'hybrid' or 'hybrid_channel'."
        assert patch_size[0]<patch_size[1], f"When upsampling, patch_size[0]<patch_size[1]."
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.ps = int(np.sqrt(projection_dim//num_channels))
        self.num_patches = [(self.img_size//patch)**2 for patch in self.patch_size]
        self.num_channels = num_channels
        self.projection_dim = projection_dim
        self.ratio = (max(self.patch_size)//min(self.patch_size))**2
        self.kernel_size = int(np.sqrt(self.ratio))
        self.final_proj_dim = self.projection_dim//self.ratio
        self.upsampling_type = upsampling_type
        # Layers
        self.proj = torch.nn.Linear(self.final_proj_dim, self.projection_dim)
        self.positions = torch.arange(start = 0,
                         end = self.num_patches[1],
                         step = 1,
                         ).to(device)
        self.position_embedding = torch.nn.Embedding(num_embeddings=self.num_patches[1],
                                                     embedding_dim = self.projection_dim,
                                                     )
        if self.upsampling_type=='hybrid':
            self.sq_patch = int(np.sqrt(self.num_patches[0]))
            self.layer = torch.nn.MaxPool2d(kernel_size = self.kernel_size, stride = self.kernel_size)
            self.seq = DoubleConv(self.projection_dim, self.projection_dim)
        elif self.upsampling_type=='hybrid_channel':
            self.sq_patch = int(np.sqrt(self.num_patches[0]))
            self.layer = torch.nn.MaxPool2d(kernel_size = self.kernel_size, stride = self.kernel_size)
            self.seq = DoubleConv(self.num_patches[1]*self.num_channels, self.num_patches[1])

    def forward(self,
                encoded_patches:torch.Tensor,
                ):
        if self.upsampling_type=='hybrid':
            encoded_patches = torch.permute(torch.reshape(encoded_patches, [-1, self.sq_patch, self.sq_patch, self.projection_dim]), [0,3,1,2])
            encoded_patches = self.layer(encoded_patches)
            encoded_patches = .5*(encoded_patches + self.seq(encoded_patches))
            encoded_patches = torch.permute(encoded_patches, [0,2,3,1])
            encoded_patches = torch.flatten(encoded_patches, start_dim = 1, end_dim = 2) + self.position_embedding(self.positions)
            return encoded_patches
        elif self.upsampling_type=='hybrid_channel':
            # Step I: MaxPool
            encoded_patches = torch.permute(torch.reshape(encoded_patches, [-1, self.sq_patch, self.sq_patch, self.projection_dim]), [0,3,1,2])
            encoded_patches = torch.flatten(torch.permute(self.layer(encoded_patches), [0,2,3,1]), start_dim = 1, end_dim = 2)
            # Step II: Conv
            encoded_patches_conv = torch.reshape(encoded_patches, [-1, self.num_patches[-1], self.num_channels, self.ps, self.ps])
            encoded_patches_conv = torch.flatten(encoded_patches_conv, start_dim = 1, end_dim = 2)
            encoded_patches_conv = self.seq(encoded_patches_conv)
            encoded_patches_conv = torch.reshape(encoded_patches, [-1, self.num_patches[-1], self.num_channels, self.ps, self.ps])
            encoded_patches_conv = torch.flatten(encoded_patches_conv, start_dim = 2, end_dim = -1)
            encoded_patches = .5*(encoded_patches + encoded_patches_conv) +  + self.position_embedding(self.positions)
            return encoded_patches
