"""Discriminator for cycle GAN.


TODO Allow for constraint network as input

"""

# Modified code from tensorflow pix2pix network under Apache license
# to support 3D network and other features.

import tensorflow as tf
from .utils import *

def discriminator(is3d=True, norm_type='instancenorm', wf=8, disc_prior=None):
    """PatchGan discriminator model similar to https://arxiv.org/abs/1611.07004.

    2d images should probably be less than 256x256.  3D volumes should be >=40x40

    Args:
        is3d: true=3d tensor; false=2d tensor
        norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
        wf: width factor.  depending on model size, might need to divide the width by a certain factor
        disc_prior: prior model will taake input of discriminator, output will be have two convolution layers after

    Returns:
        Discriminator model
    """

    initializer = tf.random_normal_initializer(0., 0.02)
    # input for model
    if is3d:
        inp = tf.keras.layers.Input(shape=[None, None, None, 1], name='input_image') # 40
    else:
        inp = tf.keras.layers.Input(shape=[None, None, 1], name='input_image')
    x = inp

    # create a simple model with a few downsample layers or reuse a provided model
    # 3 downsamples (conv + strided conv downsample)
    down, _ = downsample("1", 1, 64//wf, is3d, norm_type=norm_type, apply_norm=False) # 18
    down1 = down(x)

    ##### HACK
    # valid convolution
    if is3d:
        conv = tf.keras.layers.Conv3D(
          16, 3, strides=1, kernel_initializer=initializer,
          use_bias=False)(down1) # 16 
    else:
        conv = tf.keras.layers.Conv2D(
          16, 3, strides=1, kernel_initializer=initializer,
          use_bias=False)(x) 

    down1 = tf.keras.layers.LeakyReLU()(conv)
    #leaky_relu = tf.keras.layers.LeakyReLU()(conv)
    ##### END HACK

    down, _ = downsample("2", 128//wf, 256//wf, is3d, norm_type=norm_type) # 6 (hack)
    down2 = down(down1)
    x = down2
    dims = 32

    if disc_prior is not None:
        # reuse model
        x2 = disc_prior(inp) 
        x = tf.keras.layers.Concatenate()([x, x2])
        dims = 64

        #down, _ = downsample("p1", 32, 32, is3d, norm_type=norm_type, apply_norm=False) # 1
        #x2 = down(x2)


    down, _ = downsample("3", dims, 32, is3d, norm_type=norm_type, apply_norm=False) # 18
    x = down(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # valid convolution
    if is3d:
        conv = tf.keras.layers.Conv3D(
          256//wf, 1, strides=1, kernel_initializer=initializer,
          use_bias=False)(x)  # 1
    else:
        conv = tf.keras.layers.Conv2D(
          256//wf, 1, strides=1, kernel_initializer=initializer,
          use_bias=False)(x) 

    """
    if norm_type.lower() == 'batchnorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv)
    elif norm_type.lower() == 'instancenorm':
        norm1 = InstanceNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)
    """
    leaky_relu = tf.keras.layers.LeakyReLU()(conv)

    if is3d:
        last = tf.keras.layers.Conv3D(
          1, 1, strides=1,
          kernel_initializer=initializer)(leaky_relu)  # 1
    else:
        last = tf.keras.layers.Conv2D(
          1, 1, strides=1,
          kernel_initializer=initializer)(leaky_relu)

    return tf.keras.Model(inputs=inp, outputs=last)

