import torch
from model.functions import Patch, Unflatten, Unpatch, DownSampling, UpSampling

# Class PatchEncoder, to include initial and positional encoding
class OriginalPatchEncoder(torch.nn.Module):
    def __init__(self,
                 num_patches:int,
                 patch_size:int,
                 projection_dim:int,
                 dtype:torch.dtype,
                 ):
        super(OriginalPatchEncoder, self).__init__()
        # Parameters
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.dtype = dtype
        self.positions = torch.arange(start = 0,
                         end = self.num_patches,
                         step = 1,
                         )

        # Layers
        self.Flatten = torch.nn.Flatten(start_dim = -3,
                                        end_dim = -1,
                                        )
        self.Linear = torch.nn.Linear(3*self.patch_size**2, self.projection_dim)
        self.position_embedding = torch.nn.Embedding(num_embeddings=self.num_patches,
                                                     embedding_dim = self.projection_dim,
                                                     )

    def forward(self, patches):
        flat_patches = self.Flatten(patches)
        flat_proj_patches = self.Linear(flat_patches)
        encoded = flat_proj_patches + self.position_embedding(self.positions)
        return encoded


# AutoEncoder implementation
class FeedForward(torch.nn.Module):
    def __init__(self,
                 projection_dim:int,
                 hidden_dim:int,
                 dropout:float,
                 dtype:torch.dtype,
                 ):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(projection_dim, hidden_dim, dtype = dtype),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, projection_dim, dtype = dtype),
            torch.nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class ReAttention(torch.nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 expansion_ratio = 3,
                 apply_transform=True,
                 transform_scale=False,
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.apply_transform = apply_transform
        
        self.scale = qk_scale or head_dim ** -0.5
        if apply_transform:
            self.reatten_matrix = torch.nn.Conv2d(self.num_heads,self.num_heads, 1, 1)
            self.var_norm = torch.nn.BatchNorm2d(self.num_heads)
            self.qconv2d = torch.nn.Conv2d(3,3,3,padding = 'same', bias=qkv_bias)
            self.kconv2d = torch.nn.Conv2d(3,3,3,padding = 'same', bias=qkv_bias)
            self.vconv2d = torch.nn.Conv2d(3,3,3,padding = 'same', bias=qkv_bias)
            self.reatten_scale = self.scale if transform_scale else 1.0
        else:
            self.qconv2d = torch.nn.Conv2d(3,3,3,padding = 'same', bias=qkv_bias)
            self.kconv2d = torch.nn.Conv2d(3,3,3,padding = 'same', bias=qkv_bias)
            self.vconv2d = torch.nn.Conv2d(3,3,3,padding = 'same', bias=qkv_bias)
        
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)
    def forward(self,
                X:torch.Tensor,
                ):
        B, N, C = X.shape
        q = torch.flatten(torch.stack([self.qconv2d(y) for y in Unflatten(X)], dim = 0), -3,-1)
        k = torch.flatten(torch.stack([self.kconv2d(y) for y in Unflatten(X)], dim = 0), -3,-1)
        v = torch.flatten(torch.stack([self.vconv2d(y) for y in Unflatten(X)], dim = 0), -3,-1)
        qkv = torch.cat([q,k,v], dim = -1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (torch.matmul(q, k.transpose(-2, -1))) * self.scale
        attn = torch.nn.functional.softmax(attn, dim = -1)
        attn = self.attn_drop(attn)
        if self.apply_transform:
            attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale
        attn_next = attn
        x = (torch.matmul(attn, v)).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_next


class ReAttentionTransformerEncoder(torch.nn.Module):
    def __init__(self,
                 num_patches:int,
                 projection_dim:int,
                 hidden_dim:int,
                 num_heads:int,
                 attn_drop:int,
                 proj_drop:int,
                 linear_drop:float,
                 dtype:torch.dtype,
                 ):
        super().__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.linear_drop = linear_drop
        self.dtype = dtype
        self.ReAttn = ReAttention(self.projection_dim,
                                  num_heads = self.num_heads,
                                  attn_drop = self.attn_drop,
                                  proj_drop = self.proj_drop,
                                  )
        self.LN = torch.nn.LayerNorm(normalized_shape = (self.num_patches, self.projection_dim),
                                     dtype = self.dtype,
                                     )
        self.FeedForward = FeedForward(projection_dim = self.projection_dim,
                                       hidden_dim = self.hidden_dim,
                                       dropout = self.linear_drop,
                                       dtype = self.dtype,
                                       )
    def forward(self, encoded_patches):
        encoded_patch_attn, _ = self.ReAttn(encoded_patches)
        encoded_patches += encoded_patch_attn
        encoded_patches = self.LN(encoded_patches)
        encoded_patches += self.FeedForward(encoded_patches)
        encoded_patches = self.LN(encoded_patches)
        return encoded_patches


#Model
class ViT_model(torch.nn.Module):
    def __init__(self,
                 num_encoders:int,
                 n_classes:int,
                 num_patches:int,
                 patch_size:int,
                 projection_dim:int,
                 hidden_dim:int,
                 num_heads:int,
                 attn_drop:float,
                 proj_drop:float,
                 linear_drop:float,
                 dtype:torch.dtype,
                 ):
        super().__init__()
        # Parameters
        self.num_encoders = num_encoders
        self.n_classes = n_classes
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.linear_drop = linear_drop
        self.dtype = dtype
        # Layers
        self.OPE = OriginalPatchEncoder(self.num_patches,self.patch_size,self.projection_dim,self.dtype)
        self.Encoders = torch.nn.ModuleList()
        for _ in range(self.num_encoders):
            self.Encoders.append(
                ReAttentionTransformerEncoder(self.num_patches,
                                              self.projection_dim,
                                              self.hidden_dim,
                                              self.num_heads,
                                              self.attn_drop,
                                              self.proj_drop,
                                              self.linear_drop,
                                              self.dtype,
                                              )
            )
        self.Flat = torch.nn.Flatten()
        self.Lin1 = torch.nn.Linear(self.projection_dim*self.num_patches,128)
        self.Drop = torch.nn.Dropout(self.linear_drop)
        self.Lin2 = torch.nn.Linear(128,self.n_classes)
    
    def forward(self,X):
        Y = Patch(X, self.patch_size)
        Y = self.OPE(Y)
        for layer in self.Encoders:
            Y = layer(Y)
        Y = self.Flat(Y)
        Y = self.Lin1(Y)
        Y = self.Drop(Y)
        Y = self.Lin2(Y)
        return Y