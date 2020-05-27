"""General utitlities using the predicted network.
"""

from .datasets.generators import volume3d_ng 
from .datasets.datasets import create_dataset_from_generator, unstandardize_population
import numpy as np


def predict_ng_cube(location, start, size, model, meanstd_x, meanstd_y):
    # chunk in cubes with overlap
    rois = []
    index = []
    for xiter in range(start[0], size[0], model.outdimsize):
        for yiter in range(start[1], size[1], model.outdimsize):
            for ziter in range(start[2], size[2], model.outdimsize):
                rois.append((xiter-model.buffer, yiter-model.buffer, ziter-model.buffer))
                index.append((xiter - start[0], yiter - start[1], ziter - start[2]))

    # create dataset
    dataset = volume3d_ng(location, None, size=(model.outdimsizesize + model.buffer*2), array=rois)  
    dataset = create_dataset_from_generator(dataset, None, batch_size=1, epoch_size=len(rois), global_adjust=False, meanstd=meanstd_x)

    # run inference over all the small subvolumes
    result = model.predict(dataset)

    # populate result
    out_buffer = np.zeros(size, np.uint8)

    idx = 0
    for data_y in result:
        data_y = unstandardize_population(data_y, meanstd_y)
        out_buffer[index[idx]] = data_y[0, :, :, :, 0].numpy()
        idx += 1

    return out_buffer
