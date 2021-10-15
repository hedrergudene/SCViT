import tensorflow as tf
import numpy as np
from typing import List
from deep_vit_macula.model.tf.functions import *

# Models
## Original ViT
class ViT(tf.keras.layers.Layer):
    def __init__(self,
                 img_size:int=128,
                 patch_size:int=16,
                 num_channels:int=1,
                 num_heads:int=8,
                 transformer_layers:int=8,
                 hidden_unit_factor:float=2.,
                 mlp_head_units:List[int]=[2048,1024],
                 num_classes:int=4,
                 drop_attn:float=.2,
                 drop_linear:float=.4,
                 original_attn:bool=False,
                 ):
        super(ViT, self).__init__()
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.mlp_head_units = mlp_head_units
        self.num_classes = num_classes
        self.drop_attn = drop_attn
        self.drop_linear = drop_linear
        self.original_attn = original_attn
        self.num_patches = (self.img_size//self.patch_size)**2
        self.projection_dim = self.num_channels*self.patch_size**2
        self.hidden_units = int(hidden_unit_factor*self.projection_dim)
        # Layers
        self.PE = PatchEncoder(self.img_size, self.patch_size, self.num_channels)
        if self.original_attn:
            self.TB = AttentionTransformerEncoder(self.img_size,self.patch_size,self.num_channels,self.num_heads,self.transformer_layers, self.hidden_units,self.drop_attn)
        else:
            self.TB = ReAttentionTransformerEncoder(self.img_size,self.patch_size,self.num_channels,self.num_heads,self.transformer_layers, self.hidden_units, self.drop_attn)
        self.MLP = tf.keras.Sequential([tf.keras.layers.LayerNormalization(epsilon=1e-6),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dropout(self.drop_linear)])
        for i in self.mlp_head_units:
            self.MLP.add(tf.keras.layers.Dense(i))
            self.MLP.add(tf.keras.layers.Dropout(self.drop_linear))
        self.MLP.add(tf.keras.layers.Dense(self.num_classes))

    def call(self, X:tf.Tensor):
        # Patch
        encoded_patches = self.PE(X)
        # Transformer Block
        encoded_patches = self.TB(encoded_patches)
        # Classify outputs
        logits = self.MLP(encoded_patches)
        return logits


## HViT
class HViT(tf.keras.layers.Layer):
    def __init__(self,
                 img_size:int=128,
                 patch_size:List[int]=[16,8],
                 num_channels:int=1,
                 num_heads:int=8,
                 transformer_layers:List[int]=[5,5],
                 mlp_head_units:List[int]=[512,64],
                 num_classes:int=4,
                 hidden_unit_factor:float=.5,
                 drop_attn:float=.2,
                 drop_proj:float=.2,
                 drop_rs:float=.2,
                 drop_linear:float=.4,
                 trainable_rs:bool=True,
                 original_attn:bool=True,
                 ):
        super(HViT, self).__init__()
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_size_rev = self.patch_size[-1::-1]
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.mlp_head_units = mlp_head_units
        self.num_classes = num_classes
        self.drop_attn = drop_attn
        self.drop_proj = drop_proj
        self.drop_rs = drop_rs
        self.drop_linear = drop_linear
        self.trainable_rs = trainable_rs
        self.original_attn = original_attn
        self.num_patches = [(self.img_size//patch)**2 for patch in self.patch_size]
        self.projection_dim = [self.num_channels*patch**2 for patch in self.patch_size]
        self.hidden_units = [int(hidden_unit_factor*proj) for proj in self.projection_dim]
        # Layers
        ##Positional Encoding
        self.DPE = DeepPatchEncoder(self.img_size, self.patch_size, self.num_channels)
        ##Encoder
        self.Encoder = []
        self.Encoder_RS = []        
        if self.original_attn:
            for i in range(len(self.patch_size)):
                self.Encoder.append(AttentionTransformerEncoder(self.img_size,self.patch_size[i],self.num_channels,self.num_heads,self.transformer_layers[i], self.hidden_units[i],self.drop_attn,self.drop_proj))
                if (i+1)<len(self.patch_size):
                    self.Encoder_RS.append(Resampling(self.img_size, self.patch_size[i:i+2], self.num_channels, self.drop_rs, self.trainable_rs))
        else:
            for i in range(len(self.patch_size)):
                self.Encoder.append(ReAttentionTransformerEncoder(self.img_size,self.patch_size[i],self.num_channels,self.num_heads,self.transformer_layers[i], self.hidden_units[i],self.drop_attn,self.drop_proj))
                if (i+1)<len(self.patch_size):
                    self.Encoder_RS.append(Resampling(self.img_size, self.patch_size[i:i+2], self.num_channels, self.drop_rs, self.trainable_rs))
        ##MLP
        self.MLP = tf.keras.Sequential([tf.keras.layers.LayerNormalization(epsilon=1e-6),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dropout(self.drop_linear)])
        for i in self.mlp_head_units:
            self.MLP.add(tf.keras.layers.Dense(i))
            self.MLP.add(tf.keras.layers.Dropout(self.drop_linear))
        self.MLP.add(tf.keras.layers.Dense(self.num_classes))

    def call(self, X:tf.Tensor):
        # Patch
        encoded = self.DPE(X)
        # Encoder
        for i in range(len(self.patch_size)):
            encoded = self.Encoder[i](encoded)
            if (i+1)<len(self.patch_size):
                encoded = self.Encoder_RS[i](encoded)
        # MLP
        logits = self.MLP(encoded)
        return logits
