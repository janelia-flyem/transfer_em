"""General utitlities using the predicted network.
"""

from .datasets.generators import volume3d_ng 
from .datasets.datasets import create_dataset_from_generator, unstandardize_population
from .cgan import EM2EM
import tensorflow as tf
import numpy as np
import json
import os

def predict_cube_from_saved_model(location, start, size, cloudrun, model_dir, fetch_input=False):
    """Runs prediction on a saved model for a subvolume.

    This function automatically takes care of image paddding issues and stitching
    predicted regions together to create the subvolume.

    Args:
        location (str): cloud directory (only supports ng precomputed) where data is stored
        start (tuple): (x,y,z) start location
        size (tuple): (xsize, ysize, zsize) subvolume size
        cloudrun (str): location of cloud run serving source data
        model_dir (str): directory of saved model
        fetch_input (boolean): return input and output instead of just input
    Returns:
        numpy 3D input (optional), numpy3D output  
    """

    meta_path = os.path.join(model_dir, 'meta.json')
    model = tf.keras.models.load_model(model_dir, compile=False)
    meta = json.load(open(meta_path))

    outdimsize = meta["outdimsize"] 
    buffer = meta["buffer"]
    meanstd_x = meta["meanstd_x"]
    meanstd_y = meta["meanstd_y"]
    
    return predict_ng_cube(location, start, size, model, meanstd_x, meanstd_y, cloudrun, outdimsize=outdimsize, buffer=buffer, fetch_input=fetch_input)


def predict_ng_cube(location, start, size, model, meanstd_x, meanstd_y, cloudrun=None, fetch_input=False, outdimsize=None, buffer=None):
    """Predict specified subvolume from already loaded file..

    Note: this is like predict_cube_from_saved_model but the model has already be initialized.


    Args:
        location (str): cloud directory (only supports ng precomputed) where data is stored
        start (tuple): (x,y,z) start location
        size (tuple): (xsize, ysize, zsize) subvolume size
        model (cgan object): object for cgan model
        meanstd_x (tuple): mean, variance for X domain
        meanstd_y (tuple): mean, variance for Y target domain
        cloudrun (str): location of cloud run serving source data
        fetch_input (boolean): return input and output instead of just input
        outdimsize (int): allow overwrite with specific outdimsize
        buffer (int): allow overwrite with specific buffer size
    Returns:
        numpy 3D input (optional), numpy3D output  
    """
  
    if outdimsize is None:
        outdimsize = model.outdimsize

    if buffer is None:
        buffer = model.buffer

    # make sure outdimsize is a multiple of 8
    # (assume always even)
    tpad = 0
    if (outdimsize // 6) != 0:
        diff = outdimsize % 6
        outdimsize -= diff
        tpad = (diff // 2)
        buffer += tpad 

    # chunk in cubes with overlap
    rois = []
    index = []
    for xiter in range(start[0], start[0]+size[0], outdimsize):
        for yiter in range(start[1], start[1]+size[1], outdimsize):
            for ziter in range(start[2], start[2]+size[2], outdimsize):
                rois.append((xiter-buffer, yiter-buffer, ziter-buffer))
                index.append((xiter - start[0], yiter - start[1], ziter - start[2]))

    
    # create dataset
    dataset = volume3d_ng(location, None, size=(outdimsize + buffer*2), array=rois, cloudrun=cloudrun)  
    dataset, _ = create_dataset_from_generator(dataset, None, batch_size=1, epoch_size=len(rois), meanstd=meanstd_x)

    # populate result (add a buffer to be a multiple of outdimsize
    z, y, x = size[2], size[1], size[0]
    if (size[0] % outdimsize) != 0:
        x += (outdimsize - (size[0] % outdimsize))
    if (size[1] % outdimsize) != 0:
        y += (outdimsize - (size[1] % outdimsize))
    if (size[2] % outdimsize) != 0:
        z += (outdimsize - (size[2] % outdimsize))
    size_buf = (z,y,x)

    out_buffer = np.zeros(size_buf, np.uint8)
    if fetch_input:
        in_buffer = np.zeros(size_buf, np.uint8)

    idx = 0
    # run inference over all the small subvolumes
    for data_x in dataset:
        data_y = model.predict(data_x)
        data_y = (unstandardize_population(data_y, meanstd_y) + 1) * 127.5
       
        if tf.is_tensor(data_y):
            data_y = data_y.numpy()
        
        if tpad > 0:
            data_y = data_y[:, tpad:-tpad, tpad:-tpad, tpad:-tpad, :]

        # round array and make 8 bit
        data_y = np.around(data_y).astype(np.uint8)

        # index is xyz ... c-style buffer is zyx
        out_buffer[index[idx][2]: (index[idx][2]+outdimsize), index[idx][1]: (index[idx][1]+outdimsize),index[idx][0]: (index[idx][0]+outdimsize)] = data_y[0, :, :, :, 0]
        if fetch_input:
            data_x = (unstandardize_population(data_x, meanstd_x) + 1) * 127.5
            buf = data_x[0, buffer:(outdimsize+buffer), buffer:(outdimsize+buffer), buffer:(outdimsize+buffer), 0].numpy()
            in_buffer[index[idx][2]: (index[idx][2]+outdimsize), index[idx][1]: (index[idx][1]+outdimsize),index[idx][0]: (index[idx][0]+outdimsize)] = buf 
        idx += 1

    if fetch_input:
        return in_buffer[0:size[2], 0:size[1], 0:size[0]], out_buffer[0:size[2], 0:size[1], 0:size[0]]
    return out_buffer[0:size[2], 0:size[1], 0:size[0]]


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


