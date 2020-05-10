"""Discriminator for cycle GAN.


TODO Allow for constraint network as input

"""

# Modified code from tensorflow pix2pix network under Apache license
# to support 3D network and other features.

import tensorflow as tf
from models.utils import *


def discriminator(is3d=True, norm_type='instancenorm'):
    """PatchGan discriminator model similar to https://arxiv.org/abs/1611.07004.

    2d images should probably be less than 256x256.  3D volumes should be <=40x40

    Args:
        is3d: true=3d tensor; false=2d tensor
        norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.

    Returns:
        Discriminator model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    if is3d:
        inp = tf.keras.layers.Input(shape=[None, None, None, 1], name='input_image')
    else:
        inp = tf.keras.layers.Input(shape=[None, None, 1], name='input_image')
    x = inp

    # 2d (example sizes shown for starting 256x256) or 3d downsample
    down1 = downsample(64, 4, is3d, norm_type, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4, is3d, norm_type)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4, is3d, norm_type)(down2)  # (bs, 32, 32, 256)

    # valid convolution
    if is3d:
        conv = tf.keras.layers.Conv3D(
          512, 3, strides=1, kernel_initializer=initializer,
          use_bias=False)(down3)  # (bs, 30, 30, 512)
    else:
        conv = tf.keras.layers.Conv2D(
          512, 3, strides=1, kernel_initializer=initializer,
          use_bias=False)(down3)  # (bs, 30, 30, 512)

    if norm_type.lower() == 'batchnorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv)
    elif norm_type.lower() == 'instancenorm':
        norm1 = InstanceNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    if is3d:
        last = tf.keras.layers.Conv2D(
          1, 3, strides=1,
          kernel_initializer=initializer)(leaky_relu)  # (bs, 28, 28, 1)
    else:
        last = tf.keras.layers.Conv2D(
          1, 3, strides=1,
          kernel_initializer=initializer)(leaky_relu)  # (bs, 28, 28, 1)

    return tf.keras.Model(inputs=inp, outputs=last)


