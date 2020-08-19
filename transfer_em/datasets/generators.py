"""Contains some generator helper functions for retrieving image data.
"""

import requests
import numpy as np
import subprocess
import json
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE


def volume3d_dvid(dvid_server, uuid, instance, bbox, size=132, seed=None, array=None):
    """Returns a dataset based on a generator that will produce an infinite number of 3D volumes
    from DVID randomly from given bounding box or from provided list of ROIs.

    Note: only support uint8blk.

    Args:
        dvid_server (str): dvid server http address
        uuid (str): dvid uuid
        instance (str): name of grayscale instance
        bbox (tuple of tuples): ((x,y,z), (x2,y2,z2)) start and stop location for samples
        size (int): size of dimension
        array (list): list of rois
    Returns:
        tf.dataset containing uint8 3D tensors
    """

    def generator():
        # use manaul provided bboxes
        if array is not None:
            for start in array:
                yield start
        else:
            # make repeatable if a seed is set
            if seed is not None:
                tf.random.set_seed(seed)

            while True:
                #  get random starting point from bbox (x1,y1,z1) (x2,y2,z2)
                xstart = tf.random.uniform(shape=[], minval=bbox[0][0], maxval=bbox[1][0], dtype=tf.int64, seed=seed)
                ystart = tf.random.uniform(shape=[], minval=bbox[0][1], maxval=bbox[1][1], dtype=tf.int64, seed=seed)
                zstart = tf.random.uniform(shape=[], minval=bbox[0][2], maxval=bbox[1][2], dtype=tf.int64, seed=seed)
                yield (xstart, ystart, zstart)
                #yield tf.convert_to_tensor(fetch_raw_dvid(dvid_server, uuid, instance, [[xstart,ystart,zstart], [xstart+size, ystart+size, zstart+size]], session), dtype=tf.uint8)
    
    def mapper(xstart, ystart, zstart):
        session = requests.Session()
        return tf.convert_to_tensor(fetch_raw_dvid(dvid_server, uuid, instance, [[xstart,ystart,zstart], [xstart+size, ystart+size, zstart+size]], session), dtype=tf.uint8)

    def wrapper_mapper(x, y, z):
        tensor = tf.py_function(func = mapper, inp=(x,y,z), Tout = tf.uint8)
        tensor.set_shape((size, size, size))
        return tensor

    return tf.data.Dataset.from_generator(generator, output_types=(tf.int64, tf.int64, tf.int64)).map(wrapper_mapper, num_parallel_calls=AUTOTUNE) # ideally set to some concurrency that matches DVID's concurrency


def volume3d_ng(location, bbox, size=132, seed=None, array=None, cloudrun=None, sample_array=False, sample_class=False):
    """Returns a dataset based on a generator that will produce an infinite number of 3D volumes
    from ng precomputed randomly from given bounding box or from provided list of ROIs.

    Note: only support uint8blk.

    Note: this 

    Args:
        loccation (str): directory for precomputed volume (assume google bucket directories)
        bbox (tuple of tuples): ((x,y,z), (x2,y2,z2)) start and stop location for samples
        size (int): size of dimension
        array (list): list of rois
        cloudrun (str) location of cloud run service to retrieve volumes (if not
            provided use the locally installed tensorstore library)
    Returns:
        tf.dataset containing uint8 3D tensors
    """

    if cloudrun is None:
        try:
            import tensorstore as ts
        except ImportError:
            raise Exception("tensorstore not installed")

    def generator():
        if array is not None and sample_array == False:
            for start in array:
                yield start
        else:
            # make repeatable if a seed is set
            if seed is not None:
                tf.random.set_seed(seed)

            while True:
                curr_bbox = bbox
                if array is not None:
                    tarray = array
                    if sample_class:
                        cspot = tf.random.uniform(shape=[], minval=0, maxval=len(array), dtype=tf.int64, seed=seed)
                        tarray = array[cspot]
                    spot = tf.random.uniform(shape=[], minval=0, maxval=len(tarray), dtype=tf.int64, seed=seed)
                    curr_bbox = tarray[spot]

                #  get random starting point from bbox (x1,y1,z1) (x2,y2,z2)

                xstart = tf.random.uniform(shape=[], minval=curr_bbox[0][0], maxval=curr_bbox[1][0], dtype=tf.int64, seed=seed)
                ystart = tf.random.uniform(shape=[], minval=curr_bbox[0][1], maxval=curr_bbox[1][1], dtype=tf.int64, seed=seed)
                zstart = tf.random.uniform(shape=[], minval=curr_bbox[0][2], maxval=curr_bbox[1][2], dtype=tf.int64, seed=seed)
                yield (xstart, ystart, zstart)
       
    location_arr = location.split('/')
    bucket = location_arr[0]
    path = '/'.join(location_arr[1:])

    if cloudrun is None:
        # reuse tensorstore object
        dataset = ts.open({
            'driver': 'neuroglancer_precomputed',
            'kvstore': {
                'driver': 'gcs',
                'bucket': bucket,
                },
            'path': path,
            'recheck_cached_data': 'open',
            'scale_index': 0 
        }).result()
        dataset = dataset[ts.d['channel'][0]]
    else:
        import requests
        token = subprocess.check_output(["gcloud auth print-identity-token"], shell=True).decode()
        headers = {}
        headers["Authorization"] = f"Bearer {token[:-1]}"
        headers["Content-type"] = "application/json"

    #@tf.function
    def mapper(xstart, ystart, zstart):
        #xstart = xstart.numpy()
        #ystart = ystart.numpy()
        #zstart = zstart.numpy()
        if cloudrun is None:
            # read from tensorstore
            data = dataset[xstart:(xstart+size), ystart:(ystart+size), zstart:(zstart+size)].read().result()
            return tf.convert_to_tensor(data, dtype=tf.uint8)
        else:
            # read from cloud run function
            config = {"location": location, "size": [int(size), int(size), int(size)], "start": [int(xstart), int(ystart), int(zstart)]} 
            res = requests.post(cloudrun+"/volume", data=json.dumps(config), headers=headers)
            if res.status_code != 200:
                # refetch token if obsolete
                token = subprocess.check_output(["gcloud auth print-identity-token"], shell=True).decode()
                headers["Authorization"] = f"Bearer {token[:-1]}"
                res = requests.post(cloudrun+"/volume", data=json.dumps(config), headers=headers) 
            if res.status_code != 200:
                raise RuntimeError("cloud run failed")
            data = np.fromstring(res.content, dtype=np.uint8)
            data = data.reshape((size,size,size))
            #data = data.transpose((2,1,0))
            return tf.convert_to_tensor(data, dtype=tf.uint8)
            
    #@tf.function
    def wrapper_mapper(x, y, z):
        tensor = tf.py_function(func = mapper, inp=(x,y,z), Tout = tf.uint8)
        tensor.set_shape((size, size, size))
        return tensor

    return tf.data.Dataset.from_generator(generator, output_types=(tf.int64, tf.int64, tf.int64)).map(wrapper_mapper, num_parallel_calls=AUTOTUNE) # ideally set to some concurrency that matches number of parallel http calls possible


##### Helper function #####

def fetch_raw_dvid(server, uuid, instance, box_zyx, session):
    """
    Fetch raw array data from an instance that contains voxels.
    
    Note:
        Most voxels data instances do not support a 'scale' parameter, so it is not included here.
        Instead, by convention, we typically create multiple data instances with a suffix indicating the scale.
        For instance, 'grayscale', 'grayscale_1', 'grayscale_2', etc.
        (For labelarray and labelmap instances, see fetch_labelarray_voxels(), which does support scale.)
    
    Args:
        server:
            dvid server, e.g. 'emdata3:8900'
        
        uuid:
            dvid uuid, e.g. 'abc9'
        
        instance:
            dvid instance name, e.g. 'grayscale'
        box_zyx:
            The bounds of the volume to fetch in the coordinate system for the requested scale.
            Given as a pair of coordinates (start, stop), e.g. [(0,0,0), (10,20,30)], in Z,Y,X order.
            The box need not be block-aligned.
        
    Returns:
        np.ndarray
    """

    dtype=np.uint8

    box_zyx = np.asarray(box_zyx)
    assert np.issubdtype(box_zyx.dtype, np.integer), \
        f"Box has the wrong dtype.  Use an integer type, not {box_zyx.dtype}"
    assert box_zyx.shape == (2,3)

    params = {}
    
    shape_zyx = (box_zyx[1] - box_zyx[0])
    shape_str = '_'.join(map(str, shape_zyx[::-1]))
    offset_str = '_'.join(map(str, box_zyx[0, ::-1]))

    r = session.get(f'{server}/api/node/{uuid}/{instance}/raw/0_1_2/{shape_str}/{offset_str}', params=params)
    r.raise_for_status()

    if len(r.content) != np.prod(shape_zyx) * np.dtype(dtype).itemsize:
        info = fetch_instance_info(server, uuid, instance)
        typename = info["Base"]["TypeName"]
        msg = ("Buffer from DVID is the wrong length for the requested array.\n"
               "Did you pass the correct dtype for this instance?\n"
               f"Instance '{instance}' has type '{typename}', and you passed dtype={np.dtype(dtype).name}")
        raise RuntimeError(msg)

    a = np.frombuffer(r.content, dtype=dtype)
    return a.reshape(shape_zyx)




