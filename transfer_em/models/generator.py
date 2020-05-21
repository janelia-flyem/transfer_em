"""Generator for cycle GAN.

Currently only a simple U-net is provied,

TODO: add option for residual network blocks.

TODO: potentially support multi-scale generator
"""

# Modified code from tensorflow pix2pix network under Apache license
# to support 3D network and other features.


import tensorflow as tf
from .utils import *

def unet_generator(dimsize, is3d=True, norm_type='instancenorm'):
    """Modified u-net generator model based on https://arxiv.org/abs/1611.07004.

      Args:
        dimsize: length of each dimentions (must be square or cube)
        is3d: true=3d tensor; false=2d tensor
        norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.

      Returns:
        Generator model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    if is3d:
        last = tf.keras.layers.Conv3DTranspose(
            1, 4, strides=2,
            padding='same', kernel_initializer=initializer,
            activation='tanh') 
    else:
        last = tf.keras.layers.Conv2DTranspose(
            1, 4, strides=2,
            padding='same', kernel_initializer=initializer,
            activation='tanh')  # (bs, 256, 256, 3)

    concat = tf.keras.layers.Concatenate()

    if is3d:
        inputs = tf.keras.layers.Input(shape=[None, None, None, 1])
    else:
        inputs = tf.keras.layers.Input(shape=[None, None, 1])

    x = inputs

    # Downsampling through the model
    skips = []
    curr_size = dimsize
    max_width = 512
    curr_width = 64

    # keep downsampling until 1x1x1
    x = downsample(curr_width, 4, is3d, norm_type, apply_norm=False)(x)
    curr_size = (curr_size // 2) - 1
    
    skips.append((x, curr_width, curr_size))
    while curr_size > 16:
        curr_width *= 2
        if curr_width > max_width:
            curr_width = max_width
        x = downsample(curr_width, 4, is3d, norm_type)(x)
        curr_size = (curr_size // 2) - 1
        skips.append((x, curr_width, curr_size))

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    dcount = 0
    for (skip, curr_width, old_size) in skips:
        # only do drop-up for first 3 layers
        dcount += 1
        dropout = (dcount <= 3)
        x = upsample(curr_width, 4, is3d, norm_type, apply_dropout=dropout)(x)
        
        # only copy voxels within the current window size
        curr_size *= 2
        crop1 = (old_size - curr_size) // 2
        crop2 = crop1
        if ((old_size - curr_size) % 2) > 0:
            crop2 = crop1 + 1
        if is3d:
            skip_cropped = tf.keras.layers.Cropping3D(
                cropping=((crop1,crop2), (crop1,crop2), (crop1,crop2)))(skip) 
        else:
            skip_cropped = tf.keras.layers.Cropping2D(
                cropping=((crop1,crop2), (crop1,crop2)))(skip)
        x = concat([x, skip_cropped])

    # no initial convolutions are performed, so the last upsample
    # has no crop and copy
    x = last(x)
    curr_size *= 2

    return tf.keras.Model(inputs=inputs, outputs=x), curr_size


