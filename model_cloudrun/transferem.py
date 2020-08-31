"""Web server that serves out a tensorflow model
"""

import os

from flask import Flask, Response, request, make_response, abort
from google.cloud import storage
from flask_cors import CORS
import json
import logging
from PIL import Image
import pwd
import numpy as np
import tensorstore as ts
import tensorflow as tf
import traceback
import sys
import io
import threading
import gzip

app = Flask(__name__)

# TODO: Limit origin list here: CORS(app, origins=[...])
CORS(app)
logger = logging.getLogger(__name__)

# dictionary that contains model name, model, and meta
# (note: only one model stored in cache at a time)
MODEL_CACHE = None

# read environment variable for location of model bucket
MODEL_BUCKET = os.environ["MODEL_BUCKET"]

@app.route('/', methods=["POST"])
def transfer():
    """
    Run tensorflow prediction from stored model.
    
    1. Fetches model from cloud if not in cache
    2. Fetches volume requested (must be 64 aligned)
    3. Applies pre-processing
    4. Runs model
    5. Directly writes to specified bucket location

    {
        "location": "bucket and data location",
        "start": [x,y,z], # where to start reading -- should be multiple 64 from global offset
        "glbstart": [x,y,z], # for 0,0,0 offset 
        "size": [x,y,z]. # multiple of 64
        "model_name": "model:version",
        "dest": "bucket and dest location for neuroglancer"
    }
    """
    global MODEL_CACHE

    try:
        config_file  = request.get_json()

        # Strip gs:// prefix
        location = config_file["location"] # contains source and destination
        if location.startswith('gs://'):
            location = location[len('gs://'):]

        # check if size and start are multiples of 64
        start = config_file["start"] # in XYZ order
        glbstart = config_file["glbstart"] # in XYZ order
        size = config_file["size"] # in XYZ order
        if (start[0]-glbstart[0]) % 64 != 0 or (start[1]-glbstart[1]) % 64 != 0 or (start[2]-glbstart[2]) % 64 != 0:
            raise RuntimeError("size must be 64 block aligned")
        
        if size[0] % 64 != 0 or size[1] % 64 != 0 or size[2] % 64 != 0:
            raise RuntimeError("size must be 64 block aligned")

        location_arr = location.split('/')
        bucket = location_arr[0]
        path = '/'.join(location_arr[1:])
        stderr = sys.stderr

        storage_client = storage.Client()

        # fetch model if necessary and just put in a temporary file
        model = None
        meta = None
        if MODEL_CACHE is not None and (config_file["model_name"] == MODEL_CACHE["model_name"]):
            model = MODEL_CACHE["model"]
            meta = MODEL_CACHE["meta"]
        else:
            # fetch data from bucket
            name_arr = config_file["model_name"].split(":")
            m_name = name_arr[0]
            v_name = name_arr[1]
            bucket2 = storage_client.bucket(MODEL_BUCKET)
           
            # make temporary dir
            tmp_dir = "tmp_model"
            try:
                os.makedirs(tmp_dir)
            except Exception:
                pass
            prefix = m_name + "/" + v_name + "/"

            # download all contents
            for blob in storage_client.list_blobs(bucket2, prefix=prefix):
                dest = blob.name[len(prefix):]
                try:
                    destdir = os.path.dirname(dest)
                    if destdir != "":
                        os.makedirs(f"{tmp_dir}/{destdir}")
                except Exception:
                    pass
                blob.download_to_filename(f"{tmp_dir}/{dest}")

            meta = json.load(open(f"{tmp_dir}/meta.json"))
            model = tf.keras.models.load_model(tmp_dir, compile=False)
            MODEL_CACHE = {"model": model, "meta": meta, "model_name": config_file["model_name"]}

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

        x, y, z = start
        b = meta["buffer"]
        sx, sy, sz = size
        data = dataset[(x-b):(x+sx+b), (y-b):(y+sy+b), (z-b):(z+sz+b)].read(order='F').result()
        data = data.transpose((2,1,0))
            


        # preprocess
        # re-scale and standardize
        data = (data / 127.5) - 1
        data -= meta["meanstd_x"][0]
        data /= meta["meanstd_x"][1]
        data = np.expand_dims(data, axis=3)
        data = np.expand_dims(data, axis=0)

        # run prediction
        data_out = model.predict(data)

        # unnormalize data
        data_out *= meta["meanstd_y"][1]
        data_out += meta["meanstd_y"][0]
        data_out += 1
        data_out *= 127.5
        data_out = data_out.astype(np.uint8)
        data_out = np.squeeze(data_out)

        # write to google bucket
        # Strip gs:// prefix
        dest = config_file["dest"] # destination for n#
        if dest.startswith('gs://'):
            dest = dest[len('gs://')]

        dest_arr = dest.split('/')
        bucket = dest_arr[0]
        prefix = '/'.join(dest_arr[1:])
        bucket = storage_client.bucket(bucket)
        offsetz = (start[2]-glbstart[2])
        offsety = (start[1]-glbstart[1])
        offsetx = (start[0]-glbstart[0])

        NUM_THREADS = 2
        def write_blocks(thread_id):
            num = 0
            for ziter in range(0, size[2], 64):
                for yiter in range(0, size[1], 64):
                    for xiter in range(0, size[0], 64):
                        num += 1
                        if num % NUM_THREADS != thread_id:
                            continue
                        block = data_out[ziter:(ziter+64), yiter:(yiter+64), xiter:(xiter+64)]
                        blob = bucket.blob(f"{prefix}/{xiter+offsetx}-{xiter+64+offsetx}_{yiter+offsety}-{yiter+64+offsety}_{ziter+offsetz}-{ziter+64+offsetz}")
                        blob.content_encoding = "gzip"
                        blob.upload_from_string(gzip.compress(block.tobytes()), content_type="application/octet-stream")
        threads = [threading.Thread(target=write_blocks, args=(thread_id,)) for thread_id in range(NUM_THREADS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    except Exception as e:
        return Response(traceback.format_exc(), 400)
    return Response("success", 200)

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
