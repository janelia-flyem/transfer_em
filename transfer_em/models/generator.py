"""Generator for cycle GAN.

Currently only a simple U-net is provied,

TODO: add option for residual network blocks.

TODO: potentially support multi-scale generator
"""

# Modified code from tensorflow pix2pix network under Apache license
# to support 3D network and other features.


import tensorflow as tf
from .utils import *

# technically invalid sizes will still work but off-by-one problems could arise
VALID_DIMS = [74]

VALID_OUT = [40]

def unet_generator(dimsize, is3d=True, norm_type='instancenorm', wf=8):
    """Modified u-net generator model based on https://arxiv.org/abs/1611.07004.

      Args:
        dimsize: length of each dimentions (must be square or cube)
        is3d: true=3d tensor; false=2d tensor
        norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
        wf: width factor.  depending on model size, might need to divide the width by a certain factor

      Returns:
        Generator model
    """

    # dimsize must be valid at least for the generator
    # (this is also valid for discriminator)
    if dimsize not in VALID_DIMS:
        raise RuntimeError(f"{dimsize} does not allow for valid convolutions")

    initializer = tf.random_normal_initializer(0., 0.02)

    if is3d:
        inputs = tf.keras.layers.Input(shape=[None, None, None, 1])
    else:
        inputs = tf.keras.layers.Input(shape=[None, None, 1])

    x = inputs
    curr_dim = dimsize # 74

    # downsample 4 times

    # add an extra 3x3 convolution at beginning
    if is3d:
        x = tf.keras.layers.Conv3D(64//wf, 3, strides=1, kernel_initializer=initializer, use_bias=False)(x)
    else:
        x = tf.keras.layers.Conv2D(64//wf, 3, strides=1, kernel_initializer=initializer, use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    curr_dim = curr_dim - 2 # 72

    down, skip = downsample("1", 64//wf, 64//wf, is3d, apply_norm=False)
    skip0 = skip(x)
    down1 = down(x)
    curr_dim = curr_dim - 2 # 70
    skip0_dim = curr_dim
    curr_dim = (curr_dim // 2) - 1 # 34

    down, skip = downsample("2", 64//wf, 128//wf, is3d, norm_type=norm_type)
    skip1 = skip(down1)
    down2 = down(down1)
    curr_dim = curr_dim - 2 # 32
    skip1_dim = curr_dim
    curr_dim = (curr_dim // 2) - 1 # 15

    def concat(x, skip, dim_dn, dim_up):
        crop1 = (dim_dn - dim_up) // 2
        crop2 = crop1
        if ((dim_dn - dim_up) % 2) > 0:
            crop2 = crop1 + 1
        if is3d:
            skip_cropped = tf.keras.layers.Cropping3D(
                cropping=((crop1,crop2), (crop1,crop2), (crop1,crop2)))(skip) 
        else:
            skip_cropped = tf.keras.layers.Cropping2D(
                cropping=((crop1,crop2), (crop1,crop2)))(skip)
        x = tf.keras.layers.Concatenate()([x, skip_cropped])
        return x

    # upsample 2 times

    up1 = upsample("2", 128//wf, 128//wf, is3d, norm_type=norm_type, apply_dropout=True)(down2)
    curr_dim = (curr_dim - 2) * 2 # 26
    up1_cat = concat(up1, skip1, skip1_dim, curr_dim)
   
    # add an extra 3x3 convolution in between
    if is3d:
        x = tf.keras.layers.Conv3D(256//wf, 3, strides=1, kernel_initializer=initializer, use_bias=False)(up1_cat)
    else:
        x = tf.keras.layers.Conv2D(256//wf, 3, strides=1, kernel_initializer=initializer, use_bias=False)(up1_cat)
    x = tf.keras.layers.LeakyReLU()(x)
    curr_dim -= 2 # 24

    up0 = upsample("1", 256//wf, 64//wf, is3d, norm_type=norm_type, apply_dropout=True)(x)
    curr_dim = (curr_dim - 2) * 2 # 44 
    up0_cat = concat(up0, skip0, skip0_dim, curr_dim)

    # add a 1x1 convolution at the end instad? 
    if is3d:
        x = tf.keras.layers.Conv3D(128//wf, 3, strides=1, kernel_initializer=initializer, use_bias=False)(up0_cat)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv3D(1, 3, strides=1, kernel_initializer=initializer, use_bias=False)(x)
    else:
        x = tf.keras.layers.Conv2D(128//wf, 3, strides=1, kernel_initializer=initializer, use_bias=False)(up0_cat)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(1, 3, strides=1, kernel_initializer=initializer, use_bias=False)(x)
    curr_dim -= 4 # 40

    return tf.keras.Model(inputs=inputs, outputs=x), curr_dim


