import tensorflow as tf
import numpy as np
from typing import List

# Auxiliary methods
def patches(X:tf.Tensor,
          patch_size:int,
          ):
    num_patches = (X.shape.as_list()[1]//patch_size)**2
    X = tf.image.extract_patches(images=X,
                           sizes=[1, patch_size, patch_size, 1],
                           strides=[1, patch_size, patch_size, 1],
                           rates=[1, 1, 1, 1],
                           padding='VALID')
    return  tf.reshape(X, (-1, num_patches, X.shape.as_list()[-1]))

def unflatten(flattened, num_channels):
    if len(flattened.shape)==2:
        n, p = flattened.shape.as_list()
    else:
        _, n, p = flattened.shape.as_list()
    unflattened = tf.reshape(flattened, (-1, n, int(np.sqrt(p//num_channels)), int(np.sqrt(p//num_channels)), num_channels))
    return unflattened

# Layers

## DoubleConv
class DoubleConvResNet(tf.keras.layers.Layer):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, filters:int, pool_size:int, resnet:bool=False):
        super(DoubleConvResNet, self).__init__()
        self.resnet = resnet
        self.conv_1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, kernel_size = pool_size, strides = pool_size, padding = 'same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ELU(),
        ])
        if self.resnet:
            self.conv_2 = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters, kernel_size = pool_size, strides = pool_size, padding = 'same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ELU(),
                tf.keras.layers.Conv2D(filters, kernel_size = pool_size, strides = pool_size, padding = 'same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ELU(),
                ])
        else:
            self.conv_2 = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters, kernel_size = pool_size, strides = pool_size, padding = 'same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ELU(),
            ])

        

    def call(self, x):
        y = self.conv_1(x)
        z = self.conv_2(y)
        if self.resnet:
            return y+z
        else:
            return z



## Resampling
class Resampling(tf.keras.layers.Layer):
    def __init__(self,
                 img_size:int=128,
                 patch_size:List[int]=[8,16],
                 num_channels:int=3,
                 projection_dim:int=768,
                 resampling_type:str='doubleconv',
                 add_position:bool = False,
                 ):
        super(Resampling, self).__init__()
        # Validation
        assert resampling_type in ['max','conv', 'doubleconv', 'doubleconvresnet'], f"Resampling type must be either 'max', 'conv', 'doubleconvresnet' or 'doubleconv'."
        assert projection_dim is not None, f"Projection_dim must be specified."
        assert (int(np.sqrt(projection_dim//num_channels))==np.sqrt(projection_dim//num_channels)), f"Projection dim has to be a perfect square (per channel)."
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = [(self.img_size//patch)**2 for patch in self.patch_size]
        self.pool_size = int(np.sqrt(self.num_patches[0]//self.num_patches[1]))
        self.num_channels = num_channels
        self.projection_dim = projection_dim
        self.ps = int(np.sqrt(self.projection_dim//self.num_channels))
        self.resampling_type = resampling_type
        self.add_position = add_position
        # Layers
        if self.resampling_type=='max':
            self.sq_patch = int(np.sqrt(self.num_patches[0]))
            self.layer = tf.keras.layers.MaxPooling2D(pool_size = self.pool_size, strides = self.pool_size, padding = 'same')
            self.positions = tf.range(start=0, limit=self.num_patches[-1], delta=1)
            self.position_embedding = tf.keras.layers.Embedding(input_dim=self.num_patches[-1], output_dim=self.projection_dim)
        else:
            if self.resampling_type=='conv':
                self.layer = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(self.num_channels*self.num_patches[-1], kernel_size = self.pool_size, strides = self.pool_size, padding = 'same'),
                    tf.keras.layers.BatchNormalization(),
                ])
            elif self.resampling_type=='doubleconvresnet':
                self.layer = DoubleConvResNet(self.num_channels*self.num_patches[-1], self.pool_size, True)
            else:
                self.layer = DoubleConvResNet(self.num_channels*self.num_patches[-1], self.pool_size, False)
            self.linear = tf.keras.layers.Dense(self.projection_dim)
            if self.add_position:
                self.positions = tf.range(start=0, limit=self.num_patches[-1], delta=1)
                self.position_embedding = tf.keras.layers.Embedding(input_dim=self.num_patches[-1], output_dim=self.projection_dim)

    def call(self, encoded:tf.Tensor):
        if self.resampling_type=='max':
            encoded = tf.reshape(encoded, [-1, self.sq_patch, self.sq_patch, self.projection_dim])
            encoded = self.layer(encoded)
            encoded = tf.reshape(encoded, [-1, self.num_patches[-1], self.projection_dim]) + self.position_embedding(self.positions)
            return encoded
        else:
            encoded = unflatten(encoded, self.num_channels)
            encoded = tf.reshape(tf.transpose(encoded, [0,2,3,1,4]), [-1, self.ps, self.ps, self.num_patches[0]*self.num_channels])
            encoded = self.layer(encoded)
            encoded = tf.reshape(encoded, [-1, self.ps//self.pool_size, self.ps//self.pool_size, self.num_patches[-1], self.num_channels])
            encoded = tf.transpose(encoded, [0,3,1,2,4])
            encoded = tf.reshape(encoded, [-1, self.num_patches[-1], self.num_channels*(self.ps//self.pool_size)**2])
            if self.add_position:
                encoded = self.linear(encoded) + self.position_embedding(self.positions)
            else:
                encoded = self.linear(encoded)
            return encoded

## Patch Encoder
class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 img_size:int=128,
                 patch_size:int=16,
                 num_channels:int=1,
                 projection_dim:int=None,
                 ):
        super(PatchEncoder, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = (self.img_size//self.patch_size)**2
        self.projection_dim = projection_dim if projection_dim is not None else self.num_channels*self.patch_size**2
        self.projection = tf.keras.layers.Dense(units=self.projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=self.num_patches, output_dim=self.projection_dim
        )
          
    def get_config(self):
        config = super(PatchEncoder, self).get_config().copy()
        config.update({
                        'img_size':self.img_size,
                        'patch_size':self.patch_size,
                        'num_patches':self.num_patches,
                        'num_channels':self.num_channels,
                        'projection_dim':self.projection_dim,
                        'projection':self.projection,
                        'position_embedding':self.position_embedding,
                        })
        return config

    def call(self, X:tf.Tensor):
        X = patches(X, self.patch_size)
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(X) + self.position_embedding(positions)
        return encoded

## FeedForward
class FeedForward(tf.keras.layers.Layer):
    def __init__(self,
                 projection_dim:int,
                 hidden_dim:int,
                 dropout:float,
                 ):
        super(FeedForward, self).__init__()
        self.D1 = tf.keras.layers.Dense(hidden_dim)
        self.Drop1 = tf.keras.layers.Dropout(dropout)
        self.D2 = tf.keras.layers.Dense(projection_dim)
        self.Drop2 = tf.keras.layers.Dropout(dropout)

    def get_config(self):
        config = super(FeedForward, self).get_config().copy()
        config.update({
                        'D1':self.D1,
                        'Drop1':self.Drop1,
                        'D2':self.D2,
                        'Drop2':self.Drop2,
                        })
        return config

    def call(self, x):
        x = self.D1(x)
        x = tf.keras.activations.gelu(x)
        x = self.Drop1(x)
        x = self.D2(x)
        x = tf.keras.activations.gelu(x)
        x = self.Drop2(x)
        return x

## ReAttention
class ReAttention(tf.keras.layers.Layer):
    def __init__(self,
                 dim,
                 num_patches,
                 num_channels=1,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.2,
                 proj_drop=0.2,
                 apply_transform=True,
                 transform_scale=False,
                 ):
        super(ReAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_size = int(np.sqrt(dim))

        head_dim = self.dim // self.num_heads
        self.apply_transform = apply_transform
        self.scale = qk_scale or head_dim ** -0.5
        if apply_transform:
            self.reatten_matrix = tf.keras.layers.Conv2D(self.num_patches, 1)
            self.var_norm = tf.keras.layers.BatchNormalization()
            self.qconv2d = tf.keras.layers.Conv2D(self.num_channels,3,padding = 'same', kernel_initializer = tf.random_normal_initializer(0., 0.02), use_bias=qkv_bias)
            self.kconv2d = tf.keras.layers.Conv2D(self.num_channels,3,padding = 'same', kernel_initializer = tf.random_normal_initializer(0., 0.02), use_bias=qkv_bias)
            self.vconv2d = tf.keras.layers.Conv2D(self.num_channels,3,padding = 'same', kernel_initializer = tf.random_normal_initializer(0., 0.02), use_bias=qkv_bias)
            self.reatten_scale = self.scale if transform_scale else 1.0
        else:
            self.qconv2d = tf.keras.layers.Conv2D(self.num_channels,3,padding = 'same', kernel_initializer = tf.random_normal_initializer(0., 0.02), use_bias=qkv_bias)
            self.kconv2d = tf.keras.layers.Conv2D(self.num_channels,3,padding = 'same', kernel_initializer = tf.random_normal_initializer(0., 0.02), use_bias=qkv_bias)
            self.vconv2d = tf.keras.layers.Conv2D(self.num_channels,3,padding = 'same', kernel_initializer = tf.random_normal_initializer(0., 0.02), use_bias=qkv_bias)
        
        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj = tf.keras.layers.Dense(dim)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)
    
    def create_queries(self, x, letter):
        if letter=='q':
            x = unflatten(x, self.num_channels)
            x = tf.map_fn(fn=lambda y: tf.keras.activations.gelu(self.qconv2d(y)), elems=x)
        if letter == 'k':
            x = unflatten(x, self.num_channels)
            x = tf.map_fn(fn=lambda y: tf.keras.activations.gelu(self.kconv2d(y)), elems=x)
        if letter == 'v':
            x = unflatten(x, self.num_channels)
            x = tf.map_fn(fn=lambda y: tf.keras.activations.gelu(self.vconv2d(y)), elems=x)

        x = tf.reshape(x, shape=[-1, self.num_patches, self.dim])
        x = tf.reshape(x, shape = [-1, self.num_patches, self.num_heads, self.dim//self.num_heads, 1])
        x = tf.transpose(x, perm = [4,0,2,1,3])
        return x[0]

    def call(self, x, atten=None):
        _, N, C = x.shape.as_list()
        q = self.create_queries(x, 'q')
        k = self.create_queries(x, 'k')
        v = self.create_queries(x, 'v')
        attn = (tf.linalg.matmul(q, k, transpose_b = True)) * self.scale
        attn = tf.keras.activations.softmax(attn, axis = -1)
        attn = self.attn_drop(attn)
        if self.apply_transform:
            attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale
        attn_next = attn
        x = tf.reshape(tf.transpose(tf.linalg.matmul(attn, v), perm = [0,2,1,3]), shape = [-1, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_next


## Transformer Encoder
class AttentionTransformerEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 img_size:int,
                 patch_size:int,
                 num_channels:int,
                 num_heads:int,
                 transformer_layers:int,
                 projection_dim:int,
                 hidden_dim:int,
                 attn_drop:float,
                 proj_drop:float,
                 ):
        super(AttentionTransformerEncoder, self).__init__()
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = (self.img_size//self.patch_size)**2
        self.projection_dim = projection_dim if projection_dim is not None else self.num_channels*self.patch_size**2
        self.transformer_layers = transformer_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        # Layers
        self.LN1 = []
        self.LN2 = []
        self.Attn = []
        self.FF = []
        for _ in range(self.transformer_layers):
            self.LN1.append(tf.keras.layers.LayerNormalization())
            self.LN2.append(tf.keras.layers.LayerNormalization())
            self.Attn.append(
                tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads,
                                                   key_dim=self.projection_dim,
                                                   dropout=self.attn_drop,
                                                  )
            )
            self.FF.append(
                FeedForward(projection_dim = self.projection_dim,
                                       hidden_dim = self.hidden_dim,
                                       dropout = self.proj_drop,
                                       )
            )

    def call(self, encoded_patches):
        for i in range(self.transformer_layers):
            encoded_patch_attn = self.Attn[i](encoded_patches, encoded_patches)
            encoded_patches = tf.keras.layers.Add()([encoded_patch_attn, encoded_patches])
            encoded_patches = self.LN1[i](encoded_patches)
            encoded_patch_FF = self.FF[i](encoded_patches)
            encoded_patches = tf.keras.layers.Add()([encoded_patch_FF, encoded_patches])
            encoded_patches = self.LN2[i](encoded_patches)
        return encoded_patches


class ReAttentionTransformerEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 img_size:int,
                 patch_size:int,
                 num_channels:int,
                 num_heads:int,
                 transformer_layers:int,
                 projection_dim:int,
                 hidden_dim:int,
                 attn_drop:float,
                 proj_drop:float,
                 ):
        super(ReAttentionTransformerEncoder, self).__init__()
        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = (self.img_size//self.patch_size)**2
        self.projection_dim = projection_dim if projection_dim is not None else self.num_channels*self.patch_size**2
        self.transformer_layers = transformer_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        # Layers
        self.LN1 = []
        self.LN2 = []
        self.ReAttn = []
        self.FF = []
        for _ in range(self.transformer_layers):
            self.LN1.append(tf.keras.layers.LayerNormalization())
            self.LN2.append(tf.keras.layers.LayerNormalization())
            self.ReAttn.append(
                ReAttention(dim = self.projection_dim,
                                  num_patches = self.num_patches,
                                  num_channels = self.num_channels,
                                  num_heads = self.num_heads,
                                  attn_drop = self.attn_drop,
                                  )
            )
            self.FF.append(
                FeedForward(projection_dim = self.projection_dim,
                                       hidden_dim = self.hidden_dim,
                                       dropout = self.proj_drop,
                                       )
            )

    def call(self, encoded_patches):
        for i in range(self.transformer_layers):
            encoded_patch_attn, _ = self.ReAttn[i](encoded_patches)
            encoded_patches = encoded_patch_attn + encoded_patches
            encoded_patches = self.LN1[i](encoded_patches)
            encoded_patches = self.FF[i](encoded_patches) + encoded_patches
            encoded_patches = self.LN2[i](encoded_patches)
        return encoded_patches