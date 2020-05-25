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
VALID_DIMS = [132, 140, 148, 156, 164, 172, 180, 188, 196, 204, 212, 220, 228, 236, 244, 252, 260, 268, 276, 284, 292, 300, 308, 316, 324, 332, 340, 348, 356, 364, 372, 380, 388, 396, 404, 412, 420, 428, 436, 444, 452, 460, 468, 476, 484, 492, 500, 508]

VALID_OUT = [76, 84, 92, 100, 108, 116, 124, 132, 140, 148, 156, 164, 172, 180, 188, 196, 204, 212, 220, 228, 236, 244, 252, 260, 268, 276, 284, 292, 300, 308, 316, 324, 332, 340, 348, 356, 364, 372, 380, 388, 396, 404, 412, 420, 428, 436, 444, 452]

def unet_generator(dimsize, is3d=True, norm_type='instancenorm'):
    """Modified u-net generator model based on https://arxiv.org/abs/1611.07004.

      Args:
        dimsize: length of each dimentions (must be square or cube)
        is3d: true=3d tensor; false=2d tensor
        norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.

      Returns:
        Generator model
    """

    # dimsize must be valid at least for the generator
    # (this is also valid for discriminator)
    if dimsize not int VALID_DIMS:
        raise RuntimeError(f"{dimsize} does not allow for valid convolutions")

    initializer = tf.random_normal_initializer(0., 0.02)

    if is3d:
        inputs = tf.keras.layers.Input(shape=[None, None, None, 1])
    else:
        inputs = tf.keras.layers.Input(shape=[None, None, 1])

    x = inputs

    curr_dim = dimsize

    # downsample 4 time
    down, skip = downsample("1", 1, 64, is3d, apply_norm=False)
    skip0 = skip(inputs)
    down1 = down(inputs)
    curr_dim = curr_dim - 2
    skip0_dim = curr_dim
    curr_dim = (curr_dim // 2) - 1

    down, skip = downsample("2", 64, 128, is3d, norm_type=norm_type)
    skip1 = skip(down1)
    down2 = down(down1)
    curr_dim = curr_dim - 2
    skip1_dim = curr_dim
    curr_dim = (curr_dim // 2) - 1
   
    down, skip = downsample("3", 128, 256, is3d, norm_type=norm_type)
    skip2 = skip(down2)
    down3 = down(down2)
    curr_dim = curr_dim - 2
    skip2_dim = curr_dim
    curr_dim = (curr_dim // 2) - 1

    #down, skip = downsample("4", 256, 512, is3d, norm_type=norm_type)
    #skip3 = skip(down3)
    #down4 = down(down3)
    #skip3_dim = curr_dim - 2
    #curr_dim = (curr_dim // 2) - 1

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

    # upsample 4 timee
    #up3 = upsample("4", 512, 512, is3d, norm_type=norm_type, apply_dropout=True)(down4)
    #curr_dim = (curr_dim - 2) * 2 
    #up3_cat = concat(up3, skip3, skip3_dim, curr_dim)

    up2 = upsample("3", 256, 256, is3d, norm_type=norm_type, apply_dropout=True)(down3)
    #up2 = upsample("3", 1024, 256, is3d, norm_type=norm_type, apply_dropout=True)(up3_cat)
    curr_dim = (curr_dim - 2) * 2 
    up2_cat = concat(up2, skip2, skip2_dim, curr_dim)
    
    up1 = upsample("2", 512, 128, is3d, norm_type=norm_type, apply_dropout=True)(up2_cat)
    curr_dim = (curr_dim - 2) * 2 
    up1_cat = concat(up1, skip1, skip1_dim, curr_dim)
    
    up0 = upsample("1", 256, 64, is3d, norm_type=norm_type, apply_dropout=True)(up1_cat)
    curr_dim = (curr_dim - 2) * 2 
    up0_cat = concat(up0, skip0, skip0_dim, curr_dim)

    # add a 1x1 convolution ad the end instad? 
    if is3d:
        x = tf.keras.layers.Conv3D(1, 3, strides=1, kernel_initializer=initializer, use_bias=False)(up0_cat)
    else:
        x = tf.keras.layers.Conv2D(1, 3, strides=1, kernel_initializer=initializer, use_bias=False)(up0_cat)
    curr_dim -= 2

    return tf.keras.Model(inputs=inputs, outputs=x), curr_dim


