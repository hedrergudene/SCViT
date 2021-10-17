import tensorflow as tf
import numpy as np
from typing import List
from HViT_classification.model.functions import *

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
                 mlp_head_units:List[int]=[512,64],
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
                 patch_size:List[int]=[8,16,32],
                 projection_dim:int=None,
                 num_channels:int=1,
                 num_heads:int=8,
                 transformer_layers:List[int]=[4,4,4],
                 mlp_head_units:List[int]=[512,64],
                 num_classes:int=4,
                 hidden_unit_factor:float=2.,
                 drop_attn:float=.2,
                 drop_proj:float=.2,
                 drop_linear:float=.4,
                 resampling_type:str='standard',
                 original_attn:bool=True,
                 ):
        super(HViT, self).__init__()
        #Validations
        assert resampling_type in ['max', 'avg', 'standard', 'conv'], f"Resampling type must be either 'max', 'avg' 'conv' or 'standard'."
        assert len(patch_size)==len(transformer_layers), f"Each patch size must have its own number of transformer layers."
        assert all([img_size//patch==img_size/patch for patch in patch_size]), f"Patch sizes must divide image size."
        assert all([patch_size[i]<patch_size[i+1] for i in range(len(patch_size)-1)]), f"Patch sizes must be a strictly increasing sequence."
        assert (resampling_type in ["max","avg"] and projection_dim is not None) or (resampling_type in ["standard", "conv"] and projection_dim is None),f"\
            If resampling_type is in ['max', 'avg'], projection_dim must be specified. If resampling_type is 'standard', projection_dim is auto\
            matically computed."
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
        self.drop_linear = drop_linear
        self.resampling_type = resampling_type
        self.original_attn = original_attn
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
                    self.Encoder_RS.append(Resampling(self.img_size, self.patch_size[i:i+2], self.num_channels, self.projection_dim[i], self.resampling_type))
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
        if self.resampling_type in ['max', 'avg']:
            self.MLP = tf.keras.Sequential([tf.keras.layers.LayerNormalization(epsilon=1e-6),
                                            tf.keras.layers.GlobalAveragePooling1D(),
                                            ])
        elif self.resampling_type in ['conv']:
            if self.img_size//self.patch_size[-1]>2:
                self.MLP = tf.keras.Sequential([tf.keras.layers.LayerNormalization(epsilon=1e-6),
                                                tf.keras.layers.GlobalAveragePooling1D(),
                                                ])
            else:
                self.MLP = tf.keras.Sequential([tf.keras.layers.LayerNormalization(epsilon=1e-6),
                                                tf.keras.layers.Lambda(lambda y: tf.transpose(unflatten(y, self.num_channels), perm=[0,4,2,3,1])),
                                                tf.keras.layers.Conv2D(1, self.num_patches[-1]//2, strides = self.num_patches[-1]//2, padding = 'same'),
                                                tf.keras.layers.Lambda(lambda y: tf.transpose(y, perm=[0,4,2,3,1])),
                                                tf.keras.layers.Lambda(lambda y: tf.squeeze(tf.reshape(y, shape=[-1,1,self.num_patches[-1]**2//16]), axis = 1)),
                                                tf.keras.layers.LayerNormalization(epsilon=1e-6),
                                                tf.keras.layers.Dense(self.projection_dim[-1]),
                                                tf.keras.layers.Dropout(self.drop_linear),
                                                ])
        elif self.resampling_type in ['standard']:
            self.MLP = tf.keras.Sequential([tf.keras.layers.LayerNormalization(epsilon=1e-6),
                                            tf.keras.layers.Flatten(),
                                            ])

        for i in self.mlp_head_units:
            self.MLP.add(tf.keras.layers.Dense(i))
            self.MLP.add(tf.keras.layers.Dropout(self.drop_linear))
        self.MLP.add(tf.keras.layers.Dense(self.num_classes))


    def get_config(self):

        config = super(HViT, self).get_config().copy()
        config.update({
                        'img_size':self.img_size,
                        'patch_size':self.patch_size,
                        'patch_size_rev':self.patch_size_rev,
                        'num_channels':self.num_channels,
                        'num_heads':self.num_heads,
                        'transformer_layers':self.transformer_layers,
                        'mlp_head_units':self.mlp_head_units,
                        'num_classes':self.num_classes,
                        'drop_attn':self.drop_attn,
                        'drop_proj':self.drop_proj,
                        'drop_linear':self.drop_linear,
                        'resampling_type':self.resampling_type,
                        'original_attn':self.original_attn,
                        'num_patches':self.num_patches,
                        'projection_dim':self.projection_dim,
                        'hidden_units':self.hidden_units,
                        'PE':self.PE,
                        'Encoder':self.Encoder,
                        'Encoder_RS':self.Encoder_RS,
                        'MLP':self.MLP,
                        })
        return config

    
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