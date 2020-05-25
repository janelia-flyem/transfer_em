"""Helper functions for building discriminator and generator models.

TODO: implement mirrored padding layer

"""
# Modified code from tensorflow pix2pix network under Apache license.

import tensorflow as tf

class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, is3d=True, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
        self.is3d = is3d

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        axes = [1,2]
        if self.is3d:
            axes = [1,2,3]
        mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


def downsample(id, infilters, outfilters, is3d, filter_size=4, norm_type='instancenorm', apply_norm=True):
    """Downsamples an input.

    Conv2D => norm => LeakyRelu => Conv2D-down => Batchnorm => LeakyRelu

    Args:
        id: name of modeel
        infilters: number of input filters
        outfilters: number of output filters
        filter_size: filter size for downsampling
        is3d: true=3d tensor; false=2d tensor
        norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
        apply_norm: If True, adds the batchnorm layer

    Returns:
        Downsample Sequential Model
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    if is3d:
        convlayer = tf.keras.layers.Conv3D
        inp = tf.keras.layers.Input(shape=[None, None, None, infilters], name='input_image')
    else:
        convlayer = tf.keras.layers.Conv2D
        inp = tf.keras.layers.Input(shape=[None, None, infilters], name='input_image')

    if norm_type.lower() == 'batchnorm':
        normlayer = tf.keras.layers.BatchNormalization
    elif norm_type.lower() == 'instancenorm':
        normlayer = lambda x: InstanceNormalization(is3d)(x)

    # perform convolution
    conv = convlayer(outfilters, 3, strides=1, padding="valid",
            kernel_initializer=initializer, use_bias=False)(inp)
    norm1 = normlayer(conv)
    before_down = tf.keras.layers.LeakyReLU()(norm1)
    
    # perform downsample 
    conv = convlayer(outfilters, filter_size, strides=2, kernel_initializer=initializer, use_bias=False)(before_down)
    norm1 = normlayer(conv)
    last = tf.keras.layers.LeakyReLU()(norm1)
    
    return tf.keras.Model(inputs=inp, outputs=last, name=f"Downsample_{id}"), tf.keras.Model(inputs=inp, outputs=before_down)



def upsample(id, infilters, outfilters, is3d, filter_size=4, norm_type='instancenorm', apply_dropout=True):
    """Upsamples an input (returns same number of filters after doubling and halving.

    Conv2DTranspose => Batchnorm => Dropout => Relu

    Args:
        infilters: number of input filters
        outfilters: number of output filters
        filter_size: filter size
        norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
        apply_dropout: If True, adds the dropout layer

    Returns:
        Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    if is3d:
        convlayer = tf.keras.layers.Conv3D
        convlayerup = tf.keras.layers.Conv3DTranspose
        inp = tf.keras.layers.Input(shape=[None, None, None, infilters], name='input_image')
    else:
        convlayer = tf.keras.layers.Conv2D
        convlayerup = tf.keras.layers.Conv2DTranspose
        inp = tf.keras.layers.Input(shape=[None, None, infilters], name='input_image')

    if norm_type.lower() == 'batchnorm':
        normlayer = tf.keras.layers.BatchNormalization
    elif norm_type.lower() == 'instancenorm':
        normlayer = lambda x: InstanceNormalization(is3d)(x)

    # perform convolution
    conv = convlayer(outfilters*2, 3, strides=1, padding="valid",
            kernel_initializer=initializer, use_bias=False)(inp)
    res = normlayer(conv)
    before_up = tf.keras.layers.LeakyReLU()(res)
    
    # perform upsample 
    conv = convlayerup(outfilters, filter_size, strides=2, padding='same',
            kernel_initializer=initializer, use_bias=False, name="convup2")(before_up)
    res = normlayer(conv)
    if apply_dropout:
        res = tf.keras.layers.Dropout(0.5)(res)
    last = tf.keras.layers.LeakyReLU()(res)
    
    return tf.keras.Model(inputs=inp, outputs=last, name=f"Upsample_{id}")


