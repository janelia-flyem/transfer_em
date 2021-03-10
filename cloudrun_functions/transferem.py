"""Web server that has endpoints to write aligned data to cloud storage.
"""

import os

from flask import Flask, Response, request, make_response, abort
from flask_cors import CORS
import json
import logging
from PIL import Image
import pwd
import numpy as np
import tensorstore as ts
import traceback
import sys
import io

app = Flask(__name__)

# TODO: Limit origin list here: CORS(app, origins=[...])
CORS(app)
logger = logging.getLogger(__name__)


from collections import OrderedDict
class lru_cache:
    """The cache should be memory limited.
    For now, prevent caching of images over 1 MB.
    """
    def __init__(self, limit=256):
        self.cache = OrderedDict()
        self.limit = limit

    def get(self, img_key):
        if img_key not in self.cache:
            raise Exception("Not found")
        self.cache.move_to_end(img_key)
        return self.cache[img_key]

    def put(self, img_key, val):
        if len(val) > 1000000:
            return
        self.cache[img_key] = val
        self.cache.move_to_end(img_key)
        if len(self.cache) > self.limit:
            self.cache.popitem(last=False)

SLICE_CACHE = lru_cache()

#@app.route('/slice/<string:location>/<string:startstr>/<string:sizestr>/<string:format>', methods=["GET"])
#def slice(location, startstr, sizestr, format):
@app.route('/slice/<string:startstr>/<string:sizestr>/<string:format>', methods=["GET"])
def slice(startstr, sizestr, format):
    """
    Retrieve slice with specified format
    """
    try:
        start_arr = startstr.split("_")
        start = [int(start_arr[0]), int(start_arr[1]), int(start_arr[2])]
        
        size_arr = sizestr.split("_")
        size = [int(size_arr[0]), int(size_arr[1]), int(size_arr[2])]

        location = request.args.get("location")

        if size[0] != 1 and size[1] != 1 and size[2] != 1:
            return Response("one dimension must be size 1", 400)

        location_arr = location.split('/')
        bucket = location_arr[0]
        path = '/'.join(location_arr[1:])
        stderr = sys.stderr

        cache_key = f"{location}_{startstr}_{sizestr}_{format}"
        try:
            resp = SLICE_CACHE.get(cache_key)
        except Exception:
            # reuse tensorstore object
            dataset = ts.open({
                'driver': 'neuroglancer_precomputed',
                'kvstore': {
                    'driver': 'gcs',
                    'bucket': bucket,
                    },
                'path': path,
                'recheck_cached_data': 'open'
            }).result()
            dataset = dataset[ts.d['channel'][0]]

            x, y, z = start
            sx, sy, sz = size
            data = dataset[x:x+sx, y:y+sy, z:z+sz].read(order='F').result()
            
            # write 2D image to jpeg or png
            data = np.squeeze(data)
            imgByteArr = io.BytesIO()
            im = Image.fromarray(data.transpose((1,0)))
            im.save(imgByteArr, format=format)
            resp = imgByteArr.getvalue()
            SLICE_CACHE.put(cache_key, resp)
        r = make_response(resp)
        r.headers.set('Content-Type', f"image/{format}")
        return r

    except Exception as e:
        return Response(traceback.format_exc(), 400)


@app.route('/volume', methods=["POST"])
def volume():
    """
    Retrieve volume from tensorstore.
    See fetch_subvolume() function below for client-side example usage.
    """
    try:
        config_file  = request.get_json()

        # Strip gs:// prefix
        location = config_file["location"] # contains source and destination
        if location.startswith('gs://'):
            location = location[len('gs://'):]

        start = config_file["start"] # in XYZ order
        size = config_file["size"] # in XYZ order
        scale_index = config_file.get("scale_index", 0)
        use_jpeg = config_file.get("jpeg", False)

        location_arr = location.split('/')
        bucket = location_arr[0]
        path = '/'.join(location_arr[1:])
        stderr = sys.stderr

        # reuse tensorstore object
        dataset = ts.open({
            'driver': 'neuroglancer_precomputed',
            'kvstore': {
                'driver': 'gcs',
                'bucket': bucket,
                },
            'path': path,
            'recheck_cached_data': 'open',
            'scale_index': scale_index
        }).result()
        dataset = dataset[ts.d['channel'][0]]

        #      +--------------------------------------------------------------------+
        #      | A quick guide to 3D array index semantics and memory order choices |
        #      +--------------------------------------------------------------------+
        #
        # +-----------------+--------------+-----------------------------------------------+
        # | Index semantics | Memory order | Notes                                         |
        # +-----------------+--------------+-----------------------------------------------+
        # |        a[Z,Y,X] | C            | Standard for Python users. Prefer this.       |
        # |                 |              |                                               |
        # |        a[X,Y,Z] | F            | Identical memory layout to the above,         |
        # |                 |              | (the RAM contents are identical to the above) |
        # |                 |              | but due to the reverse index meaning, this    |
        # |                 |              | is likely to introduce confusion and/or       |
        # |                 |              | accidental inefficiences when you pass this   |
        # |                 |              | array to library functions which expect a     |
        # |                 |              | standard C-order array.                       |
        # |                 |              |                                               |
        # |        a[Z,Y,X] | F            | Never do this.                                |
        # |                 |              |                                               |
        # |        a[X,Y,Z] | C            | Never do this.                                |
        # +-----------------+--------------+-----------------------------------------------+

        # Unfortunately, TensorStore.read() always returns an [X,Y,Z]-indexed array,
        # but it does permit you to specify the memory ordering.
        # Therefore, we request F-order, the only sane choice.
        # The buffer we'll return to the caller can be interpreted as either F/XYZ or C/ZYX.
        # (That's their business.)

        x, y, z = start
        sx, sy, sz = size
        data = dataset[x:x+sx, y:y+sy, z:z+sz].read(order='F').result()
        if not use_jpeg or sz > 1:
            r = make_response(data.tobytes(order='F'))
            r.headers.set('Content-Type', 'application/octet-stream')
            return r
        else:
            # write 2D image to JPEG
            data = np.squeeze(data)
            imgByteArr = io.BytesIO()
            im = Image.fromarray(data.transpose((1,0)))
            im.save(imgByteArr, format="JPEG")
            r = make_response(imgByteArr.getvalue())
            r.headers.set('Content-Type', 'image/jpeg')
            return r

    except Exception as e:
        return Response(traceback.format_exc(), 400)


def fetch_subvolume(service_url, location, box_zyx, scale_index=0, dtype=None):
    """
    Example client function to fetch 3D subvolumes.

    Args:
        service_url:
            CloudRun URL, e.g. https://transferem-qdoifjasf23jk348-uk.a.run.app

        location:
            bucket location of the data, e.g. gs://mybucket/neuroglancer/jpeg

        box_zyx:
            (start_zyx, stop_xyz)
            Subvolume start/stop corners, in ZYX order

        scale_index:
            Which pyramid scale to read data from.

        dtype:
            Must match the dtype of the remote volume.
            Default is np.uint8

    Returns:
        ndarray, ZYX-indexed, C-order
        (If desired, transpose to get XYZ-index, F-order)
    """
    import numpy as np

    assert not service_url.startswith('http://'), \
        "service must start with https://"

    if not service_url.startswith('https'):
        service_url = f'https://{service_url}'

    dtype = dtype or np.uint8

    box_zyx = np.asarray(box_zyx)
    assert box_zyx.shape == (2,3), "subvolume must be 3D"

    # REST API expects XYZ order
    box_xyz = box_zyx[:, ::-1]
    shape_xyz = box_xyz[1] - box_xyz[0]

    config = {
        'location': location,
        'start': box_xyz[0].tolist(),
        'size': shape_xyz.tolist(),
        'scale_index': scale_index
    }

    r = requests.post(f'{service_url}/volume', json=config)
    r.raise_for_status()

    data = np.frombuffer(r.content, dtype=dtype)

    X, Y, Z = shape_xyz
    return data.reshape((Z, Y, X))

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
