"""General utitlities using the predicted network.
"""

from .datasets.generators import volume3d_ng 
from .datasets.datasets import create_dataset_from_generator, unstandardize_population
from .cgan import EM2EM
import tensorflow as tf
import numpy as np
import json


def predict_ng_cube(location, start, size, model, meanstd_x, meanstd_y, cloudrun=None, fetch_input=False):
    """Predict specified subvolume.

    This function automatically fetches data with proper context to predict the specified region.

    Note: start and size is specified as X, Y, Z values.
    """
    
    # chunk in cubes with overlap
    rois = []
    index = []
    for xiter in range(start[0], start[0]+size[0], model.outdimsize):
        for yiter in range(start[1], start[1]+size[1], model.outdimsize):
            for ziter in range(start[2], start[2]+size[2], model.outdimsize):
                rois.append((xiter-model.buffer, yiter-model.buffer, ziter-model.buffer))
                index.append((xiter - start[0], yiter - start[1], ziter - start[2]))

    
    # create dataset
    dataset = volume3d_ng(location, None, size=(model.outdimsize + model.buffer*2), array=rois, cloudrun=cloudrun)  
    dataset, _ = create_dataset_from_generator(dataset, None, batch_size=1, epoch_size=len(rois), meanstd=meanstd_x)

    # populate result (add a buffer to be a multiple of outdimsize
    z, y, x = size[2], size[1], size[0]
    if (size[0] % model.outdimsize) != 0:
        x += (model.outdimsize - (size[0] % model.outdimsize))
    if (size[1] % model.outdimsize) != 0:
        y += (model.outdimsize - (size[1] % model.outdimsize))
    if (size[2] % model.outdimsize) != 0:
        z += (model.outdimsize - (size[2] % model.outdimsize))
    size_buf = (z,y,x)

    out_buffer = np.zeros(size_buf, np.uint8)
    if fetch_input:
        in_buffer = np.zeros(size_buf, np.uint8)

    idx = 0
    # run inference over all the small subvolumes
    for data_x in dataset:
        data_y = model.predict(data_x)

        data_y = (unstandardize_population(data_y, meanstd_y) + 1) * 127.5

        # index is xyz ... c-style buffer is zyx
        out_buffer[index[idx][2]: (index[idx][2]+model.outdimsize), index[idx][1]: (index[idx][1]+model.outdimsize),index[idx][0]: (index[idx][0]+model.outdimsize)] = data_y[0, :, :, :, 0].numpy()
        if fetch_input:
            data_x = (unstandardize_population(data_x, meanstd_x) + 1) * 127.5
            buf = data_x[0, model.buffer:(model.outdimsize+model.buffer), model.buffer:(model.outdimsize+model.buffer), model.buffer:(model.outdimsize+model.buffer), 0].numpy()
            in_buffer[index[idx][2]: (index[idx][2]+model.outdimsize), index[idx][1]: (index[idx][1]+model.outdimsize),index[idx][0]: (index[idx][0]+model.outdimsize)] = buf 
        idx += 1

    if fetch_input:
        return in_buffer[0:size[0], 0:size[1], 0:size[2]], out_buffer[0:size[0], 0:size[1], 0:size[2]]
    return out_buffer[0:size[0], 0:size[1], 0:size[2]]


def save_model(name, ckpt_dir, meanstd_x, meanstd_y, size=132, is3d=True):
    """Save generator model for inference in google AI platform.

    Note: metadata for size and buffer is stored as a JSON.

    Args:
        name (str): Name for model
        ckpt_dir (str): Location of checkpoint including epoch number
        meanstd_x ((float, float): mean and stddev for x
        meanstd_y ((float, float): mean and stddev for y
        size (int): dimension size
        is3d (boolean): 3d or 2d model
    """
    model = EM2EM(size, name, is3d=is3d, ckpt_restore=ckpt_dir)

    tf.keras.models.save_model(
        model.generator_g,
        name,
        overwrite=True,
        include_optimizer=False,
        save_format=None,
        signatures=None,
        options=None
    )

    meta = {
        "buffer": model.buffer,
        "outdimsize": model.outdimsize,
        "meanstd_x": [float(meanstd_x[0]), float(meanstd_x[1])],
        "meanstd_y": [float(meanstd_y[0]), float(meanstd_y[1])]
    }

    fout = open(name+"/meta.json", 'w')
    fout.write(json.dumps(meta))
    fout.close()


