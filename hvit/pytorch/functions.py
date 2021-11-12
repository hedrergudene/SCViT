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

## Unpatch
def unpatch(x, num_channels):
    if len(x.size()) < 5:
        batch_size, num_patches, ch, h, w = unflatten(x, num_channels).size()
    else:
        batch_size, num_patches, ch, h, w = x.size()
    assert ch==num_channels, f"Num. channels must agree"
    elem_per_axis = int(np.sqrt(num_patches))
    patches_middle = torch.stack([torch.cat([patch for patch in x.reshape(batch_size,elem_per_axis,elem_per_axis,ch,h,w)[i]], dim = -2) for i in range(batch_size)], dim = 0)
    restored_images = torch.stack([torch.cat([patch for patch in patches_middle[i]], dim = -1) for i in range(batch_size)], dim = 0).reshape(batch_size,1,ch,h*elem_per_axis,w*elem_per_axis)
    restored_images = torch.squeeze(restored_images, dim = 1)
    return restored_images

## Downsampling
def downsampling(encoded_patches, num_channels):
    _, _, embeddings = encoded_patches.size()
    ch, h, w = num_channels, int(np.sqrt(embeddings/num_channels)), int(np.sqrt(embeddings/num_channels))
    original_image = unpatch(unflatten(encoded_patches, num_channels), num_channels)
    new_patches = patch(original_image, patch_size = h//2)
    new_patches_flattened = torch.flatten(new_patches, start_dim = -3, end_dim = -1)
    return new_patches_flattened

## Upsampling
def upsampling(encoded_patches, num_channels):
    _, _, embeddings = encoded_patches.size()
    h, w, ch = int(np.sqrt(embeddings/num_channels)), int(np.sqrt(embeddings/num_channels)), num_channels
    original_image = unpatch(unflatten(encoded_patches, num_channels), num_channels)
    new_patches = patch(original_image, patch_size = h*2)
    new_patches_flattened = torch.flatten(new_patches, start_dim = -3, end_dim = -1)
    return new_patches_flattened


# Layers

## DoubleConv
class DoubleConv(torch.nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, num_channels:int):
        super(DoubleConv, self).__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, num_channels, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(num_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_channels, num_channels, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(num_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

## Patch Encoder
class PatchEncoder(torch.nn.Module):
    def __init__(self,
                 img_size:int,
                 patch_size:int,
                 num_channels:int,
                 projection_dim:int=None,
                 dtype:torch.dtype=torch.float32,
                 ):
        super(PatchEncoder, self).__init__()
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.dtype = dtype
        self.projection_dim = projection_dim if projection_dim is not None else self.num_channels*self.patch_size**2
        self.num_patches = (self.img_size//self.patch_size)**2
        self.positions = torch.arange(start = 0,
                         end = self.num_patches,
                         step = 1,
                         )

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

## ReAttention
class ReAttention(torch.nn.Module):
    def __init__(self,
                 dim,
                 num_channels=3,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 apply_transform=True,
                 transform_scale=False,
                 ):
        super().__init__()
        # Parameters
        self.num_heads = num_heads
        self.num_channels = num_channels
        head_dim = dim // num_heads
        self.apply_transform = apply_transform
        self.scale = qk_scale or head_dim ** -0.5

        # Layers
        if apply_transform:
            self.reatten_matrix = torch.nn.Conv2d(self.num_heads,self.num_heads, 1, 1)
            self.var_norm = torch.nn.BatchNorm2d(self.num_heads)
            self.qconv2d = torch.nn.Conv2d(self.num_channels,self.num_channels,3,padding = 'same', bias=qkv_bias)
            self.kconv2d = torch.nn.Conv2d(self.num_channels,self.num_channels,3,padding = 'same', bias=qkv_bias)
            self.vconv2d = torch.nn.Conv2d(self.num_channels,self.num_channels,3,padding = 'same', bias=qkv_bias)
            self.reatten_scale = self.scale if transform_scale else 1.0
        else:
            self.qconv2d = torch.nn.Conv2d(self.num_channels,self.num_channels,3,padding = 'same', bias=qkv_bias)
            self.kconv2d = torch.nn.Conv2d(self.num_channels,self.num_channels,3,padding = 'same', bias=qkv_bias)
            self.vconv2d = torch.nn.Conv2d(self.num_channels,self.num_channels,3,padding = 'same', bias=qkv_bias)
        
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x, atten=None):
        B, N, C = x.shape
        q = torch.flatten(torch.stack([self.qconv2d(y) for y in unflatten(x, self.num_channels)], dim = 0), -3,-1).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        k = torch.flatten(torch.stack([self.kconv2d(y) for y in unflatten(x, self.num_channels)], dim = 0), -3,-1).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        v = torch.flatten(torch.stack([self.vconv2d(y) for y in unflatten(x, self.num_channels)], dim = 0), -3,-1).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        attn = (torch.matmul(q, k.transpose(-2, -1))) * self.scale
        attn = torch.nn.functional.softmax(attn, dim = -1)
        attn = self.attn_drop(attn)
        if self.apply_transform:
            attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale
        attn_next = attn
        x = (torch.matmul(attn, v)).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

## TransformerEncoder
class TransformerEncoder(torch.nn.Module):
    def __init__(self,
                 img_size:int,
                 patch_size:int,
                 num_channels:int,
                 projection_dim:int=None,
                 hidden_dim_factor:float=2.,
                 num_heads:int=8,
                 attn_drop:float=.2,
                 proj_drop:float=.2,
                 linear_drop:float=.2,
                 original_attn:bool=True,
                 ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (self.img_size//self.patch_size)**2
        self.num_channels = num_channels
        self.projection_dim = projection_dim if projection_dim is not None else self.num_channels*self.patch_size**2
        self.hidden_dim_factor = hidden_dim_factor
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.linear_drop = linear_drop
        if original_attn:
            self.Attn = torch.nn.MultiheadAttention(projection_dim, self.num_heads, self.attn_drop, batch_first = True)
        else:
            self.Attn = ReAttention(self.projection_dim,
                                    num_channels = self.num_channels,
                                    num_heads = self.num_heads,
                                    qkv_bias = True,
                                    attn_drop = self.attn_drop,
                                    proj_drop = self.proj_drop,
                                    )
        self.LN1 = torch.nn.LayerNorm(normalized_shape = (self.num_patches, self.projection_dim),
                                     )
        self.LN2 = torch.nn.LayerNorm(normalized_shape = (self.num_patches, self.projection_dim),
                                     )
        self.FeedForward = FeedForward(projection_dim = self.projection_dim,
                                       hidden_dim_factor = self.hidden_dim_factor,
                                       dropout = self.linear_drop,
                                       )
    def forward(self, encoded_patches):
        encoded_patch_attn = self.Attn(encoded_patches)
        encoded_patches = encoded_patch_attn + encoded_patches
        encoded_patches = self.LN1(encoded_patches)
        encoded_patches = self.FeedForward(encoded_patches) + encoded_patches
        encoded_patches = self.LN2(encoded_patches)
        return encoded_patches

## Upsampling
class Upsampling(torch.nn.Module):
    def __init__(self,
                 img_size:int,
                 patch_size:List[int],
                 num_channels:int,
                 projection_dim:int=768,
                 upsampling_type:str='conv',
                 ):
        super(Upsampling, self).__init__()
        # Validation
        assert upsampling_type in ['max', 'conv'], f"Upsampling type must either be 'max' or 'conv'."
        assert patch_size[0]<patch_size[1], f"When upsampling, patch_size[0]<patch_size[1]."
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.ps = int(np.sqrt(projection_dim//num_channels))
        self.num_patches = [(self.img_size//patch)**2 for patch in self.patch_size]
        self.num_channels = num_channels
        self.projection_dim = [projection_dim for patch in self.patch_size]
        self.ratio = (max(self.patch_size)//min(self.patch_size))**2
        self.kernel_size = int(np.sqrt(self.ratio))
        self.final_proj_dim = self.projection_dim[0]//self.ratio
        self.standard_proj_dim = self.projection_dim[0]*self.ratio
        self.upsampling_type = upsampling_type
        # Layers
        self.proj = torch.nn.Linear(self.final_proj_dim, self.projection_dim[1])
        if self.upsampling_type=='conv':
            self.sequence = torch.nn.Sequential(
                torch.nn.Conv2d(self.num_channels*self.num_patches[0], self.num_channels*self.num_patches[1], kernel_size = self.kernel_size, stride = self.kernel_size),
                DoubleConv(self.num_channels*self.num_patches[1]),
            )
        else:
            self.sequence = torch.nn.MaxPool2d(kernel_size, stride=kernel_size)

    def forward(self,
                encoded_patches:torch.Tensor,
                ):
        if self.upsampling_type=='conv':
            encoded_patches = unflatten(encoded_patches, self.num_channels)
            encoded_patches = torch.flatten(encoded_patches, start_dim = 1, end_dim = 2)
            encoded_patches = self.sequence(encoded_patches)
            encoded_patches = encoded_patches.reshape((-1, self.num_patches[1], self.num_channels, self.ps//2, self.ps//2))
            encoded_patches = encoded_patches.flatten(-3, -1)
            encoded_patches = self.proj(encoded_patches)
            return encoded_patches
        elif self.upsampling_type=='max':
            encoded_patches = encoded.permute((0,2,1)).reshape((-1, self.projection_dim[0], int(np.sqrt(self.num_patches[0])), int(np.sqrt(self.num_patches[0]))))
            encoded_patches = sequence(encoded_patches)
            encoded_patches = encoded_patches.flatten(-2,-1).permute((0,2,1))
            return encoded_patches
