import tensorflow as tf
import numpy as np
from typing import List
from .functions import *

# Models
## Original ViT
class ViT(tf.keras.layers.Layer):
    def __init__(self,
                 img_size:int=128,
                 patch_size:int=16,
                 projection_dim:int=None,
                 num_channels:int=1,
                 num_heads:int=8,
                 transformer_layers:int=12,
                 hidden_unit_factor:float=2.,
                 mlp_head_units:List[int]=[256,64],
                 num_classes:int=4,
                 drop_attn:float=.2,
                 drop_proj:float=.2,
                 drop_linear:float=.2,
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
        self.drop_proj = drop_proj
        self.drop_linear = drop_linear
        self.num_patches = (self.img_size//self.patch_size)**2
        self.projection_dim = projection_dim if projection_dim is not None else self.num_channels*self.patch_size**2
        self.hidden_units = int(hidden_unit_factor*self.projection_dim)
        # Layers
        self.PE = PatchEncoder(self.img_size, self.patch_size, self.num_channels, self.projection_dim)
        self.TB =  AttentionTransformerEncoder(self.img_size,
                                               self.patch_size,
                                               self.num_channels,
                                               self.num_heads,
                                               self.transformer_layers,
                                               self.projection_dim, 
                                               self.hidden_units,
                                               self.drop_attn,
                                               self.drop_proj,
                                               )
        self.MLP = tf.keras.Sequential([tf.keras.layers.LayerNormalization(epsilon=1e-6),
                                        tf.keras.layers.GlobalAveragePooling1D()])
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

    
## SCViT
class SCViT(tf.keras.layers.Layer):
    def __init__(self,
                 img_size:int=32,
                 patch_size:List[int]=[2,4,8,16],
                 projection_dim:int=192,
                 num_channels:int=3,
                 num_heads:int=4,
                 transformer_layers:List[int]=[6,6,6,6],
                 hidden_unit_factor:float=2.,
                 drop_attn:float=.05,
                 drop_proj:float=.05,
                 drop_linear:float=.25,
                 original_attn:bool=True,
                 add_position:bool=True,
                 ):
        super(SCViT, self).__init__()
        #Validations
        assert len(patch_size)==len(transformer_layers), f"Each patch size must have its own number of transformer layers."
        assert all([img_size//patch==img_size/patch for patch in patch_size]), f"Patch sizes must divide image size."
        assert all([patch_size[i]<patch_size[i+1] for i in range(len(patch_size)-1)]), f"Patch sizes must be a strictly increasing sequence."
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.drop_attn = drop_attn
        self.drop_proj = drop_proj
        self.drop_linear = drop_linear
        self.resampling_type = "maxconv"
        self.original_attn = original_attn
        self.add_position = add_position
        self.num_patches = [(self.img_size//patch)**2 for patch in self.patch_size]
        self.projection_dim = [projection_dim if projection_dim is not None else self.num_channels*patch**2 for patch in self.patch_size]
        self.hidden_units = [int(hidden_unit_factor*proj) for proj in self.projection_dim]
        # Layers
        ##Positional Encoding
        self.PE = PatchEncoder(self.img_size, self.patch_size[0], self.num_channels, self.projection_dim[0])
        ##Encoder
        self.Encoder = []
        self.Encoder_RS = []        
        if self.original_attn:
            for i in range(len(self.patch_size)):
                self.Encoder.append(    
                                    AttentionTransformerEncoder(self.img_size,
                                                                self.patch_size[i],
                                                                self.num_channels,
                                                                self.num_heads,
                                                                self.transformer_layers[i],
                                                                self.projection_dim[i], 
                                                                self.hidden_units[i],
                                                                self.drop_attn,
                                                                self.drop_proj,
                                                                )
                                        )

                if (i+1)<len(self.patch_size):
                    self.Encoder_RS.append(Resampling(self.img_size, self.patch_size[i:i+2], self.num_channels, self.projection_dim[i], self.resampling_type, self.add_position))
        else:
            for i in range(len(self.patch_size)):
                self.Encoder.append(
                                    ReAttentionTransformerEncoder(self.img_size,
                                                                  self.patch_size[i],
                                                                  self.num_channels,
                                                                  self.num_heads,
                                                                  self.transformer_layers[i],
                                                                  self.projection_dim[i],
                                                                  self.hidden_units[i],
                                                                  self.drop_attn,
                                                                  self.drop_proj,
                                                                  )
                                        )
                if (i+1)<len(self.patch_size):
                    self.Encoder_RS.append(Resampling(self.img_size, self.patch_size[i:i+2], self.num_channels, self.projection_dim[i], self.resampling_type))
        ##MLP
        self.MLP = tf.keras.Sequential([tf.keras.layers.LayerNormalization(epsilon=1e-6),
                                            tf.keras.layers.GlobalAveragePooling1D(),
                                            ])

    
    def call(self, X:tf.Tensor):
        # Patch
        encoded = self.PE(X)
        # Encoder
        for i in range(len(self.patch_size)):
            encoded = self.Encoder[i](encoded)
            if (i+1)<len(self.patch_size):
                encoded = self.Encoder_RS[i](encoded)
        # MLP
        logits = self.MLP(encoded)
        return logits
