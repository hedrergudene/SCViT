import tensorflow as tf
import numpy as np
from typing import List
from HViT_classification.model.functions import *

class HViT_UNet(tf.keras.layers.Layer):
    def __init__(self,
                 img_size:int=128,
                 patch_size:List[int]=[8,16,32],
                 projection_dim:int=None,
                 num_channels:int=3,
                 num_heads:int=8,
                 transformer_layers:List[int]=[4,4],
                 size_bottleneck:int=4,
                 hidden_unit_factor:float=2.,
                 drop_attn:float=.2,
                 drop_proj:float=.2,
                 resampling_type:str='conv',
                 ):
        super(HViT_UNet, self).__init__()
        #Validations
        assert resampling_type in ['max', 'avg', 'conv', 'standard'], f"Resampling type must be either 'max', 'avg', 'conv' or 'standard'."
        assert all([img_size//patch==img_size/patch for patch in patch_size]), f"Patch sizes must divide image size."
        assert all([patch_size[i]<patch_size[i+1] for i in range(len(patch_size)-1)]), f"Patch sizes must be a strictly increasing sequence."
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_size_rev = self.patch_size[-1::-1]
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.size_bottleneck = size_bottleneck
        self.drop_attn = drop_attn
        self.drop_proj = drop_proj
        self.resampling_type = resampling_type
        self.num_patches = [(self.img_size//patch)**2 for patch in self.patch_size]
        if projection_dim is not None:
            self.projection_dim = [projection_dim for _ in self.patch_size]
        else:
            self.projection_dim = [self.num_channels*patch**2 for patch in self.patch_size]
        self.hidden_units = [int(hidden_unit_factor*proj) for proj in self.projection_dim]
        # Layers
        ## Pre-Post processing
        self.preprocessing = tf.keras.layers.Conv2D(self.num_channels, 7, 1, padding = 'same')
        self.final_linear = tf.keras.layers.Dense(self.num_channels*self.patch_size[0]**2)
        self.postprocessing = tf.keras.layers.Conv2D(self.num_channels, 7, 1, padding = 'same')
        ##Positional Encoding
        self.PE = PatchEncoder(self.img_size, self.patch_size[0], self.num_channels, self.projection_dim[0])
        ##Encoder
        self.Encoder = []
        self.Encoder_RS = []        
        for i in range(len(self.patch_size)-1):
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
            self.Encoder_RS.append(
                                    Resampling(self.img_size,
                                                self.patch_size[i:i+2],
                                                self.num_channels,
                                                self.projection_dim[i+1],
                                                self.resampling_type,
                                                )
                                    )
        ##BottleNeck
        self.BottleNeck = tf.keras.Sequential([
                                                AttentionTransformerEncoder(self.img_size,
                                                                            self.patch_size[i],
                                                                            self.num_channels,
                                                                            self.num_heads,
                                                                            self.size_bottleneck,
                                                                            self.projection_dim[-1], 
                                                                            self.hidden_units[-1],
                                                                            self.drop_attn,
                                                                            self.drop_proj,
                                                                            )
                                                ])
        ##Decoder
        self.Decoder = []
        self.Decoder_RS = []         
        for i in range(len(self.patch_size)-1):
            self.Decoder_RS.append(
                                    Resampling(self.img_size,
                                                self.patch_size_rev[i:i+2],
                                                self.num_channels,
                                                self.projection_dim[len(patch_size)-(i+2)],
                                                'standard',
                                                )
                            )
            self.Decoder.append(
                                AttentionTransformerEncoder(self.img_size,
                                                            self.patch_size_rev[i+1],
                                                            self.num_channels,
                                                            self.num_heads,
                                                            self.transformer_layers[len(patch_size)-(i+2)],
                                                            self.projection_dim[len(patch_size)-(i+2)], 
                                                            self.hidden_units[len(patch_size)-(i+2)],
                                                            self.drop_attn,
                                                            self.drop_proj,
                                                            )
                                )

        ## Skip connections
        self.SkipConnections = []
        for i in range(len(self.patch_size)-1):
            self.SkipConnections.append(
                                        SkipConnection(self.img_size,
                                                       self.patch_size_rev[i+1],
                                                       self.num_channels,
                                                       self.projection_dim[len(patch_size)-(i+2)],
                                                       self.num_heads,
                                                       self.drop_attn,
                                                       )
                                        )

    def call(self, X:tf.Tensor):
        # Preprocessing
        X_prep = self.preprocessing(X)
        # Patch
        encoded = self.PE(X_prep)
        # Encoder
        encoded_list = []
        for i in range(len(self.patch_size)-1):
            encoded = self.Encoder[i](encoded)
            encoded_list.append(encoded)
            encoded = self.Encoder_RS[i](encoded)
        # BottleNeck
        encoded = self.BottleNeck(encoded)
        # Decoder
        encoded_list = encoded_list[-1::-1]
        for i in range(len(self.patch_size)-1):
            encoded = self.Decoder_RS[i](encoded)
            encoded = self.Decoder[i](encoded)
            encoded = self.SkipConnections[i](encoded_list[i], encoded)
        # Return original image
        encoded = self.final_linear(encoded)
        Y = X + tf.squeeze(unpatch(unflatten(encoded, self.num_channels), self.num_channels), axis = 1)
        output = self.postprocessing(Y)
        return output
