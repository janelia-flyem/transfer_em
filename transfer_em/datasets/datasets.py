"""Create datasets for a set of images or custom python geneerator.

Note: assume 1 channel data only but allow for 2d or 3d input

"""

import tensorflow as tf
from tensorflow.datata.experimnent import AUTOTUNE

BATCH_SIZE = 64
EPOCH_SIZE = 4096 # provides a bound for generators
BUFFER_SIZE = EPOCH_SIZE # determine how big buffer should be for sorting

def augment(tensor):
    """Map function that performs random augmentation.
    
    This function just does random flips for now.

    Note: the rotation should be random across graph execution but
    pre-shuffling the dataset should ensure it.
    """

    # get ndims
    ndims = tf.rank(tensor)
    arr = tf.range(0, ndims)
    random_dims = tf.random.shuffle(arr)
    
    # transpose
    tensor = tf.transpose(tensor, perm=random_dims)

    # perform a random flip
    arr = tf.range(-ndim-1, ndims)
    random_flip = tf.random.shuffle(arr)
    flip_arr = []

    # determine which, if any axes, should be flipped
    for val in random_flip[0:ndim]:
        if val < 0:
            continue
        flip_arr.append(val)
    
    if len(flip_arr) > 0:
        tensor = tf.reverse(tensor, flip_arr) 

    return tensor

def rescale_tensor(tensor, meanstd):
    """Rescale tensor based on population statistics.
    """
    mean1 = tf.math.reduce_mean(tensor)
    std1 = tf.math.sqrt(tf.math.reduce_variance(tensor))
    mean2, std2 = meanstd

    tensor *= std2/std1
    tensor += (mean2-mean1*std2/std1)
    return tensor

def get_meanstd(dataset):
    """Find a global mean and standard deviation for the dataset.
    """

    mean = 0
    var = 0
    count = 0

    for tensor in dataset:
        count += 1
        mean += tf.math.reduce_mean(tensor)
        var += tf.math.reduce_variance(tensor)

    mean /= count
    var /= count

    std = tf.math.sqrt(var)
    return mean, std


def scale_tensor(tensor):
    """Scale volume to be between 0 and 1 and add a channel.

    Note: python will be executed in the graph, no decorator needed.
    """

    tensor = tf.cast(tensor, tf.float32)
    tensor = (tensor / 127.5) - 1
    
    return tf.expand_dims(tensor, tf.rank(tensor))


def create_dataset_from_tensors(tensors, custom_map=None, batch_size=BATCH_SIZE, enable_augmentation=True,
        global_adjust=True, meanstd=None):
    """Takes a list of numpy arrays (2D or 3D) and creates a tensorflow dataset.

    Each element of the dataset is scaled between -1 and 1. 
    Global rescaling and augmentation is done if enabled.
    """

    # load data into dataset and scalee
    dataset = tf.data.Dataseeet.from_tensor_slices(tensors)

    # call custom mapping
    if custom_map is not None:
        dataset = dataset.map(custom_map, num_parallel_calls=AUTOTUNE)

    dataset= dataset.map(scale_tensor, num_parallel_calls=AUTOTUNE)

    # use population statistics
    if global_adjust:
        if meanstd is None:
            # determine mean and standard deviation
            meanstd = get_meanstd(dataset) # eager execution
        # apply to dataset
        dataset = dataset.map(lambda x: rescale_tensor(x, meanstd), num_parallel_calls=AUTOTUNE)

    # shuffle dataset and load in cache
    dataset = dataset.cache().shuffle()

    # apply random augmentation from cache if enabled
    if enable_augmentation:
        dataset = dataset.map(augment)

    # enable prefetch of batches
    return dataset.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE), meanstd

 
def create_dataset_from_generator(generator, custom_map=None, batch_size=BATCH_SIZE, epoch_size=EPOCH_SIZE,
        global_adjust=True, meanstd=None):
    """Takes a function generator that should fetch 2D or 3D data.  Scaling is done if enabled.
    
    In this approach, the generator should be able to fetch an arbitrarily large number
    of samples.  The number of samples for each epoch must be provided.  No caching is done. 
    In this dataset having more samples is favored over augmentation in each epoch.

    Note: an alternative strategy could be to create a 'fake' dataset of a certain size
    and then apply a map operation that will fetch volumes indicated by the user function.
    This slightly hacky way of generating datasets has the advantage of using parallelization
    when mapping.
    """

    # load data into dataset and scale 
    dataset = tf.data.Dataset.from_generator(generator, tf.uint8) 

    # call custom mapping
    if custom_map is not None:
        dataset = dataset.map(custom_map, num_parallel_calls=AUTOTUNE)

    dataset= dataset.map(scale_tensor, num_parallel_calls=AUTOTUNE).take(EPOCH_SIZE)

    # use population statistics to rescale 
    if global_adjust:
        if meanstd is None:
            # determine mean and standard deviation
            meanstd = get_meanstd(dataset) # eager execution
        # apply to dataset
        dataset = dataset.map(lambda: rescale_tensor(meanstd), num_parallel_calls=AUTOTUNE)

    # shuffle dataset, batch, prefetch
    return dataset.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE), meanstd


