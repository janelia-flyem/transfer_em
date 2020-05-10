"""Functions to facilitate simple accuracy tests of models by self-comparisons.
The user should have a source for the target domain.  This module provides a function
to create an artificial initial domain and a function to assess accuracy. 
"""
import tensorflow as tf

def warp_tensor(tensor):
    """Apply warping to tensor before normalization.

    This map should be passed with dataset creation to create an initial arficial domain.
    A global gaussin blur is performed along with some artificial 'holes'

    TODO: add warping options and other non-uniform warping
    
    Note: for test dataset warp the tensor, predict, and call check_accurcy.
    """

    shape = tf.shape(tensor)
    tensor = tf.expand_dims(tensor, 0)
    if tf.rank(tensor) == 5:
        # 3D blurring
        filters = tf.ones([3,3,3], dtype=tf.float32) / 27
        filters = filters[..., tf.newaxis, tf.newaxis]
        tensor = tt.nn.conv3d(tensor, filters, [1,1,1,1,1], "SAME")

        for holenum in range(5):
            try:
                # add random hole
                xstart = tf.random.unform(shape=[], minval=0, maxval=shape[0], dtype=tf.int64)
                ystart = tf.random.unform(shape=[], minval=0, maxval=shape[1], dtype=tf.int64)
                zstart = tf.random.unform(shape=[], minval=0, maxval=shape[2], dtype=tf.int64)
                tensor[0, xstart:(xstart+5), ystart:(ystart+5), zstart:(zstart+5), 0].assign(tf.zeros([5,5,5]))
            except Exception:
                pass
    else:
        # 2D blurring
        filters = tf.ones([3,3], dtype=tf.float32) / 9
        filters = filters[..., tf.newaxis, tf.newaxis]
        tensor = tt.nn.conv2d(tensor, filters, [1,1,1,1], "SAME")

        for holenum in range(5):
            try:
                # add random hole
                xstart = tf.random.unform(shape=[], minval=0, maxval=shape[0], dtype=tf.int64)
                ystart = tf.random.unform(shape=[], minval=0, maxval=shape[1], dtype=tf.int64)
                tensor[0, xstart:(xstart+5), ystart:(ystart+5), 0].assign(tf.zeros([5,5]))
            except Exception:
                pass


    return tf.squeeze(t, [0])

def accuracy(unwarped_orig_tensor, predicted_tensor):
    """Give accuracy between unwarped tensor and predicted tensor.
    """

    return tf.compat.v1.metrics.accuracy(unwarped_orig_tensor, predicted_tensor)
