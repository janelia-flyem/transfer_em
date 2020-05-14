"""Functions to facilitate simple accuracy tests of models by self-comparisons.
The user should have a source for the target domain.  This module provides a function
to create an artificial initial domain and a function to assess accuracy. 
"""
import tensorflow as tf

def warp_tensor(tensor):
    """Apply warping to tensor after scaling between -1 and 1.

    This map should be passed with dataset creation to create an initial arficial domain.
    A global gaussin blur is performed along with some artificial 'holes'

    TODO: add warping options and other non-uniform warping
    
    Note: for test dataset warp the tensor, predict, and call check_accurcy.
    """

    #tf.config.run_functions_eagerly(True)

    num_hole_rate = 4 / (128*128) # percent of selected pixels in downsample imagee

    tensor = tf.expand_dims(tensor, 0)
    if tensor.shape.rank == 5:
        # 3D blurring
        filters = tf.ones([3,3,3], dtype=tf.float32) / 27
        filters = filters[..., tf.newaxis, tf.newaxis]
        tensor = tf.nn.conv3d(tensor, filters, [1,1,1,1,1], "SAME")

        for holenum in range(num_holes):
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
        tensor = tf.nn.conv2d(tensor, filters, [1,1,1,1], "SAME")

        # make hole
        uniform_random = tf.random.uniform([tensor.shape[1]*tensor.shape[2]], 0, 1.0)
        uniform_random = tf.reshape(uniform_random, tensor.shape)
        mask_matrix = tf.where(uniform_random < num_hole_rate, tf.ones_like(tensor), tf.zeros_like(tensor)) 
    
        # dilate holes 
        filters = tf.ones([4,4], dtype=tf.float32) 
        filters = filters[..., tf.newaxis, tf.newaxis]
        mask_matrix = tf.nn.conv2d(mask_matrix, filters, [1,1,1,1], "SAME")
        
        # apply mask 
        tensor = tf.where(mask_matrix > 0, -tf.ones_like(tensor), tensor)   

    return tf.squeeze(tensor, [0])

def accuracy(unwarped_orig_tensor, predicted_tensor):
    """Give accuracy between unwarped tensor and predicted tensor.
    """

    return tf.compat.v1.metrics.accuracy(unwarped_orig_tensor, predicted_tensor)
