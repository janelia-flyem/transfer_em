"""Create datasets for a set of images or custom python geneerator.

Note: assume 1 channel data only but allow for 2d or 3d input

"""

import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 64
EPOCH_SIZE = 4096 # provides a bound for generators
BUFFER_SIZE = EPOCH_SIZE # determine how big buffer should be for sorting

def create_dataset_from_tensors(tensors, custom_map=None, batch_size=BATCH_SIZE, enable_augmentation=True,
        global_adjust=True, meanstd=None, randomize=False, padding=None):
    """Takes a list of numpy arrays (2D or 3D) and creates a tensorflow dataset.

    Each element of the dataset is scaled between -1 and 1. 
    Global rescaling and augmentation is done if enabled.

    Args:
        tensors (list): list of 2d or 3d uint8 numpy arrays
        custom_map (func): custom map routine performed after scaling the tensor to between -1 and 1
        batch_size (int): number of tensors in a batch (probably use only 1 for 3D volumes)
        enable_augmentation (boolean): if true, the tensors will be randomly flipped for each epoch
        global_adjut (boolean): if true, the tensors will be scaled by a global or specified mean/stddev
        meanstd (tuple): mean, stddev if specified is used for scaling, otherwise computer from set of tensors
        randomize (boolean): shuffle the tensors
        padding (int): adds reflection padding of given size if specified

    Returns:
        4D or 5D tensor (batch, ..., ch) with batches determined by batch size
    """
    
    # load data into dataset
    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    
    # reflection pad
    if padding is not None:
        dataset = dataset.map(lambda x: tf.pad(x, padding, "REFLECT"), num_parallel_calls=AUTOTUNE)

    # scale
    dataset = dataset.map(scale_tensor, num_parallel_calls=AUTOTUNE)

    # call custom mapping
    if custom_map is not None:
        dataset = dataset.map(custom_map, num_parallel_calls=AUTOTUNE)

    # use population statistics
    if global_adjust:
        if meanstd is None:
            # determine mean and standard deviation
            meanstd = get_meanstd(dataset) # eager execution
        # apply to dataset
        dataset = dataset.map(lambda x: standardize_population(x, meanstd), num_parallel_calls=AUTOTUNE)

    # shuffle dataset and load in cache
    dataset = dataset.cache()
    if randomize:
        dataset = dataset.shuffle(BUFFER_SIZE)

    # apply random augmentation from cache if enabled
    if enable_augmentation:
        dataset = dataset.map(augment)

    # enable prefetch of batches
    return dataset.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE), meanstd

def create_dataset_from_generator(dataset, shape, custom_map=None, batch_size=BATCH_SIZE, epoch_size=EPOCH_SIZE,
        global_adjust=True, meanstd=None, padding=None):
    """Takes a dataset from a function generator that should fetch 2D or 3D data.  Scaling is done if enabled.
    
    In this approach, the generator should be able to fetch an arbitrarily large number
    of samples (infinite).  The number of samples for each epoch must be provided.  No caching is done. 
    In this dataset having more samples is favored over augmentation in each epoch.

    Each element of the dataset is scaled between -1 and 1. 

    Args:
        dataset (tf.dataset): dataset produced by a generator
        shape (tuple): Deprecated and ignored
        custom_map (func): custom map routine performed after scaling the tensor to between -1 and 1
        batch_size (int): number of tensors in a batch (probably use only 1 for 3D volumes)
        epoch_size (int): number of tensors for an epoch (must be set because the generator stream could be infinite)
        global_adjut (boolean): if true, the tensors will be scaled by a global or specified mean/stddev
        meanstd (tuple): mean, stddev if specified is used for scaling, otherwise computer from set of tensors
        padding (int): adds reflection padding of given size if specified

    Returns:
        4D or 5D tensor (batch, ..., ch) with batches determined by batch size
    """
    
    # reflection pad
    if padding is not None:
        dataset = dataset.map(lambda x: tf.pad(x, padding, "REFLECT"), num_parallel_calls=AUTOTUNE)

    # load data into dataset and scale 
    dataset = dataset.map(scale_tensor, num_parallel_calls=AUTOTUNE)
    

    # call custom mapping
    if custom_map is not None:
        dataset = dataset.map(custom_map, num_parallel_calls=AUTOTUNE)

    dataset= dataset.take(epoch_size)

    # use population statistics to standardize to the population 
    if global_adjust:
        if meanstd is None:
            # determine mean and standard deviation
            meanstd = get_meanstd(dataset) # eager execution
        # apply to dataset
        dataset = dataset.map(lambda x: standardize_population(x, meanstd), num_parallel_calls=AUTOTUNE)

    # shuffle dataset, batch, prefetch
    return dataset.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE), meanstd

###### Helper functions #######

def augment(tensor):
    """Map function that performs random augmentation.
    
    This function just does random flips for now.

    Note: the rotation should be random across graph execution but
    pre-shuffling the dataset should ensure it.
    """

    # get ndims
    ndims = tensor.shape.rank - 1 # don't include channel
    arr = tf.range(0, ndims)
    random_dims = tf.random.shuffle(arr)
    channel = tf.constant(ndims, shape=[1])
    random_dims = tf.concat([random_dims, channel], 0)

    # transpose
    tensor = tf.transpose(tensor, perm=random_dims)

    # perform a random flip
    for dim in range(ndims):
        uniform_random = tf.random.uniform([], 0, 1.0)
        if tf.math.less(uniform_random, .5):
            tensor = tf.reverse(tensor, [dim])
        
    return tensor

def standardize_population(tensor, meanstd):
    """Standardize tensor based on population statistics.
    """
    mean, std = meanstd
    tensor -= mean
    tensor /= std
    return tensor

def unstandardize_population(tensor, meanstd):
    """Undo standardization.
    """
    mean, std = meanstd
    tensor *= std
    tensor += mean
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
    
    return tf.expand_dims(tensor, tensor.shape.rank)



