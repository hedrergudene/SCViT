import numpy as np
import torch
from typing import List
import itertools


# Auxiliary functions to create & undo patches
def patch(X:torch.Tensor,
          patch_size:int,
          ):
    if len(X.size())==5:
        X = torch.squeeze(X, dim=1)
    h, w = X.shape[-2], X.shape[-1]
    assert h%patch_size==0, f"Patch size must divide images height"
    assert w%patch_size==0, f"Patch size must divide images width"
    patches = X.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patch_list = torch.flatten(patches, 2,3).permute(0,2,1,3,4)
    return patch_list

def unflatten(flattened, num_channels):
        # Alberto: Added to reconstruct from bs, n, projection_dim -> bs, n, c, h, w
        bs, n, p = flattened.size()
        unflattened = torch.reshape(flattened, (bs, n, num_channels, int(np.sqrt(p//num_channels)), int(np.sqrt(p//num_channels))))
        return unflattened

def unpatch(x, num_channels):
    if len(x.size()) < 5:
        batch_size, num_patches, ch, h, w = unflatten(x, num_channels).size()
    else:
        batch_size, num_patches, ch, h, w = x.size()
    assert ch==num_channels, f"Num. channels must agree"
    elem_per_axis = int(np.sqrt(num_patches))
    patches_middle = torch.stack([torch.cat([patch for patch in x.reshape(batch_size,elem_per_axis,elem_per_axis,ch,h,w)[i]], dim = -2) for i in range(batch_size)], dim = 0)
    restored_images = torch.stack([torch.cat([patch for patch in patches_middle[i]], dim = -1) for i in range(batch_size)], dim = 0).reshape(batch_size,1,ch,h*elem_per_axis,w*elem_per_axis)
    return restored_images


# Auxiliary methods to downsampling & upsampling
def resampling(encoded_patches:torch.Tensor,
               patch_size:List[int]=[16,8],
               num_channels:int=1,
               ):
    original_image = unpatch(unflatten(encoded_patches, num_channels), num_channels)
    new_patches = patch(original_image, patch_size = patch_size[1])
    new_patches_flattened = torch.flatten(new_patches, start_dim = -3, end_dim = -1)
    return new_patches_flattened


# Class PatchEncoder, to include initial and positional encoding
class PatchEncoder(torch.nn.Module):
    def __init__(self,
                 img_size:int=128,
                 patch_size:List[int]=[16,8],
                 num_channels:int=1,
                 device:str='cuda:0',
                 ):
        super(PatchEncoder, self).__init__()
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = [(self.img_size//patch)**2 for patch in self.patch_size]
        self.projection_dim = [self.num_channels*patch**2 for patch in self.patch_size]
        self.device = device
        if self.patch_size[0]>self.patch_size[1]:
            self.positions = torch.arange(start = 0,
                             end = self.num_patches[1],
                             step = 1,
                             device = self.device,
                             )
        else:
            self.positions = torch.arange(start = 0,
                             end = self.num_patches[0],
                             step = 1,
                             device = self.device,
                             )
        # Layers
        self.linear = torch.nn.Linear(self.projection_dim[0], self.projection_dim[0])
        self.position_embedding = torch.nn.Embedding(num_embeddings=self.num_patches_final,
                                                     embedding_dim = self.num_channels*self.patch_size_final**2,
                                                     device = self.device,
                                                     )

    def forward(self, X):
        if self.patch_size[0]>self.patch_size[1]:
            patches = patch(X, self.patch_size[1])
            flat_patches = torch.flatten(patches, -3, -1)
            encoded = flat_patches + self.position_embedding(self.positions)
            encoded = unflatten(encoded, self.num_channels)
            encoded = unpatch(encoded, self.num_channels)
            encoded = torch.flatten(patch(encoded, patch_size = self.patch_size[0]), -3, -1)
            encoded = self.linear(encoded)
            return encoded
        else:
            patches = patch(X, self.patch_size[0])
            flat_patches = torch.flatten(patches, -3, -1)
            encoded = flat_patches + self.position_embedding(self.positions)
            encoded = self.linear(encoded)
            return encoded


# AutoEncoder implementation
class FeedForward(torch.nn.Module):
    def __init__(self,
                 projection_dim:int,
                 hidden_dim_factor:float=2.,
                 dropout:float=.2,
                 device:str = 'cuda:0',
                 ):
        super().__init__()
        self.device = device
        self.net = torch.nn.Sequential(
            torch.nn.Linear(projection_dim, int(hidden_dim_factor*projection_dim), device = self.device),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(int(hidden_dim_factor*projection_dim), projection_dim, device = self.device),
            torch.nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class ReAttention(torch.nn.Module):
    def __init__(self,
                 dim,
                 num_channels=3,
                 num_heads=8,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.2,
                 proj_drop=0.2,
                 apply_transform=True,
                 transform_scale=False,
                 device:str='cuda:0',
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.num_channels = num_channels
        head_dim = dim // num_heads
        self.apply_transform = apply_transform
        self.scale = qk_scale or head_dim ** -0.5
        self.device = device
        if apply_transform:
            self.reatten_matrix = torch.nn.Conv2d(self.num_heads,self.num_heads, 1, 1, device = self.device)
            self.var_norm = torch.nn.BatchNorm2d(self.num_heads, device = self.device)
            self.qconv2d = torch.nn.Conv2d(self.num_channels,self.num_channels,3,padding = 'same', bias=qkv_bias, device = self.device)
            self.kconv2d = torch.nn.Conv2d(self.num_channels,self.num_channels,3,padding = 'same', bias=qkv_bias, device = self.device)
            self.vconv2d = torch.nn.Conv2d(self.num_channels,self.num_channels,3,padding = 'same', bias=qkv_bias, device = self.device)
            self.reatten_scale = self.scale if transform_scale else 1.0
        else:
            self.qconv2d = torch.nn.Conv2d(self.num_channels,self.num_channels,3,padding = 'same', bias=qkv_bias, device = self.device)
            self.kconv2d = torch.nn.Conv2d(self.num_channels,self.num_channels,3,padding = 'same', bias=qkv_bias, device = self.device)
            self.vconv2d = torch.nn.Conv2d(self.num_channels,self.num_channels,3,padding = 'same', bias=qkv_bias, device = self.device)
        
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim, device = self.device)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x:torch.Tensor):
        B, N, C = x.shape
        q = torch.flatten(torch.stack([self.qconv2d(y) for y in unflatten(x, self.num_channels)], dim = 0), -3,-1).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        k = torch.flatten(torch.stack([self.kconv2d(y) for y in unflatten(x, self.num_channels)], dim = 0), -3,-1).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        v = torch.flatten(torch.stack([self.vconv2d(y) for y in unflatten(x, self.num_channels)], dim = 0), -3,-1).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        attn = (torch.matmul(q, k.transpose(-2, -1))) * self.scale
        attn_soft = torch.nn.functional.softmax(attn, dim = -1)
        attn_drop = self.attn_drop(attn_soft)
        if self.apply_transform:
            attn_norm = self.var_norm(self.reatten_matrix(attn_drop)) * self.reatten_scale
        attn_next = attn_norm
        x = (torch.matmul(attn_norm, v)).transpose(1, 2).reshape(B, N, C)
        x_proj = self.proj(x)
        x_drop = self.proj_drop(x_proj)
        return x_drop, attn_next


class ReAttentionTransformerBlock(torch.nn.Module):
    def __init__(self,
                 img_size:int=128,
                 patch_size:int=16,
                 num_channels:int=1,
                 hidden_dim_factor:float=2.,
                 num_heads:int=8,
                 attn_drop:int=.2,
                 proj_drop:int=.2,
                 linear_drop:float=.2,
                 transformer_blocks:int=5,
                 device:str='cuda:0',
                 ):
        super().__init__()
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = (self.img_size//self.patch_size)**2
        self.projection_dim = self.num_channels*self.patch_size**2
        self.hidden_dim_factor = hidden_dim_factor
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.linear_drop = linear_drop
        self.transformer_blocks = transformer_blocks
        self.device = device
        # Layers
        self.ReAttn = torch.nn.ModuleList()
        self.LN1 = torch.nn.ModuleList()
        self.LN2 = torch.nn.ModuleList()
        self.FF = torch.nn.ModuleList()
        for _ in range(self.transformer_layers):
            self.ReAttn.append(ReAttention(self.projection_dim,
                                           num_channels = self.num_channels,
                                           num_heads = self.num_heads,
                                           attn_drop = self.attn_drop,
                                           proj_drop = self.proj_drop,
                                           )
                                )
            self.LN1.append(torch.nn.LayerNorm(normalized_shape = (self.num_patches, self.projection_dim),
                                     device = self.device,
                                     )
                            )
            self.LN2.append(torch.nn.LayerNorm(normalized_shape = (self.num_patches, self.projection_dim),
                                     device = self.device,
                                     )
                            )
            self.FF.append(FeedForward(projection_dim = self.projection_dim,
                                       hidden_dim_factor = self.hidden_dim_factor,
                                       dropout = self.linear_drop,
                                       )
                            )            


    def forward(self, encoded_patches):
        for i in range(self.transformer_layers):
            encoded_patch_attn, _ = self.ReAttn[i](encoded_patches)
            encoded_patches = encoded_patch_attn + encoded_patches
            encoded_patches = self.LN1[i](encoded_patches)
            encoded_patches = self.FF[i](encoded_patches) + encoded_patches
            encoded_patches = self.LN2[i](encoded_patches)
        return encoded_patches


# Model architecture
class ViT_model(torch.nn.Module):
    def __init__(self,
                 img_size:int=128,
                 patch_size:List[int]=[16,8],
                 transformer_blocks:List[int]=[5,5],
                 num_channels:int=1,
                 hidden_dim_factor:float=2.,
                 num_heads:int=8,
                 attn_drop:int=.2,
                 proj_drop:int=.2,
                 linear_list:List[int]=[2048,1024],
                 linear_drop:float=.4,
                 device:str='cuda:0',
                 ):
        super().__init__()
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.transformer_blocks = transformer_blocks
        self.num_channels = num_channels
        self.num_patches = [(self.img_size//patch)**2 for patch in self.patch_size]
        self.projection_dim = [self.num_channels*(patch)**2 for patch in self.patch_size]
        self.hidden_dim_factor = hidden_dim_factor
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.linear_list = linear_list
        self.linear_drop = linear_drop
        self.device = device
        # Layers
        self.PE = PatchEncoder(self.img_size,self.patch_size,self.num_channels)
        self.Encoders = torch.nn.ModuleList()
        for i in range(len(self.patch_size)):
            self.Encoders.append(ReAttentionTransformerBlock(self.img_size,
                                                             self.patch_size[i],
                                                             self.num_channels,
                                                             self.hidden_dim_factor,
                                                             self.num_heads,
                                                             self.attn_drop,
                                                             self.proj_drop,
                                                             self.linear_drop,
                                                             self.transformer_blocks[i],
                                                             self.device,
                                                             )
                                )

        # Output
        self.Tube = torch.nn.ModuleList()
        self.linear_list = [self.num_channels*self.num_patches*self.patch_size**2] + self.linear_list
        for i in range(len(self.linear_list)-1):
            self.Tube.append(torch.nn.Linear(in_features=self.linear_list[i], out_features=self.linear_list[i+1], dtype = self.dtype, device = self.device))
            self.Tube.append(torch.nn.Dropout(self.linear_drop))

    def forward(self,
                X:torch.Tensor,
                ):
        # "Preprocessing"
        X_patch = self.PE(X)
        # Encoders
        for i, enc in enumerate(self.Encoders):
            X_patch = enc(X_patch)
            if i<len(self.Encoders):
                X_patch = resampling(X_patch, self.patch_size[i:i+1],self.num_channels)
        # Output
        X_flat = torch.flatten(X_patch, 1, -1)
        for _, tube in enumerate(self.Tube):
            X_flat = tube(X_flat)
        return X_flat