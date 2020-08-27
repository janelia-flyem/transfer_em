"""Function for handling Google AI Platform prediction calls.

Based on example from https://cloud.google.com/ai-platform/prediction/docs/custom-prediction-routines.
"""

import tensorflow as tf
import os
import json
from google.cloud import storage
import requests       
import numpy as np
import subprocess
import random
import string
import gzip
import threading
#import google.auth
#import google.auth.transport.requests


TEMP_NG_DIR = "flyem_public_cache"

# TODO:
# Note: Custom does not work with GPU so is somewhat limited currently
# Make authenticated cloud run calls
# Fix authorization issues for cloud storage writing

class TransferEMPredictor(object):
    """Prediction utility from tranfering EM to another EM domain.
    """

    def __init__(self, model, meta, model_dir):
        """Stores artifacts for prediction. Only initialized via `from_path`.
        """
        self._model = model
        self.buffer = meta["buffer"]
        self.meanstd_x = meta["meanstd_x"]
        self.meanstd_y = meta["meanstd_y"]
        self.model_dir = model_dir

    def predict(self, instances, **kwargs):
        """Performs custom prediction.

        Accepts bounding box request, fetches data, scales images,
        perform prediction, re-scales, and converts to base64.

        Args:
            instances: A list of prediction input instances (only one instance allowed).
            **kwargs: A dictionary of keyword args provided as additional
                fields on the predict request body.

        Returns:
            A list of outputs containing the prediction results.
        """

        try:
            if len(instances) != 1:
                raise RuntimeError("only one instance allowed")
        
            location = instances[0]["location"]
            cloudrun = instances[0]["cloudrun"]
            start = tuple(instances[0]["start"])
            size = tuple(instances[0]["size"])

            # grab token without gcloud command
            #creds, project = google.auth.default()
            #auth_req = google.auth.transport.requests.Request()
            #creds.refresh(auth_req)
            #token = creds.token
            #token = subprocess.check_output(["gcloud auth print-identity-token"], shell=True).decode()
            
            headers = {}
            headers["Authorization"] = f"Bearer {token[:-1]}"
            headers["Content-type"] = "application/json" 

            config = {"location": location, "size": [size[0]+2*self.buffer, size[1]+2*self.buffer, size[2]+2*self.buffer], "start": [start[0]-self.buffer, start[1]-self.buffer, start[2]-self.buffer]}
            res = requests.post(cloudrun+"/volume", data=json.dumps(config), headers=headers)
            data = np.frombuffer(res.content, dtype=np.uint8) 
            data = data.reshape((size[2]+2*self.buffer,size[1]+2*self.buffer,size[0]+2*self.buffer))

            # re-scale and standardize
            data = (data / 127.5) - 1
            data -= self.meanstd_x[0]
            data /= self.meanstd_x[1]
            data = np.expand_dims(data, axis=3)
            data = np.expand_dims(data, axis=0)

            # run prediction
            data_out = self._model.predict(data)

            # unnormalize data
            data_out *= self.meanstd_y[1]
            data_out += self.meanstd_y[0]
            data_out += 1
            data_out *= 127.5
            data_out = data_out.astype(np.uint8)
            data_out = np.squeeze(data_out)

            # write to google bucket
            storage_client = storage.Client()
            bucket = storage_client.bucket(TEMP_NG_DIR)

            # create random name
            letters = string.ascii_lowercase
            random_dir = ''.join(random.choice(letters) for i in range(20))

            # write config
            config = {
                            "@type" : "neuroglancer_multiscale_volume",
                            "data_type" : "uint8",
                            "num_channels" : 1,
                            "scales" : [
                                {
                                    "chunk_sizes" : [
                                        [ 64, 64, 64 ]
                                        ],
                                    "encoding" : "raw",
                                    "key" : "8.0x8.0x8.0",
                                    "resolution" : [ 8,8,8 ],
                                    "size" : [ size[0], size[1], size[2] ],
                                    "offset": [0, 0, 0]
                                }
                            ],
                            "type" : "image"
                        }
            blob = bucket.blob(random_dir + "/info")
            blob.upload_from_string(json.dumps(config))
            prefix = random_dir + "/8.0x8.0x8.0"

            NUM_THREADS = 4
            def write_blocks(thread_id):
                num = 0
                for ziter in range(0, size[2], 64):
                    for yiter in range(0, size[1], 64):
                        for xiter in range(0, size[0], 64):
                            num += 1
                            if num % NUM_THREADS != thread_id:
                                continue
                            block = data_out[ziter:(ziter+64), yiter:(yiter+64), xiter:(xiter+64)]
                            blob = bucket.blob(f"{prefix}/{xiter}-{xiter+64}_{yiter}-{yiter+64}_{ziter}-{ziter+64}")
                            blob.content_encoding = "gzip"
                            blob.upload_from_string(gzip.compress(block.tobytes()), content_type="application/octet-stream")

            threads = [threading.Thread(target=write_blocks, args=(thread_id,)) for thread_id in range(NUM_THREADS)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            return [f"https://neuroglancer-demo.appspot.com/#!%7B%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%7B%22url%22%3A%22precomputed%3A%2F%2Fgs%3A%2F%2F{TEMP_NG_DIR}%2F{random_dir}%22%7D%2C%22tab%22%3A%22source%22%2C%22name%22%3A%22jpeg%22%7D%5D%2C%22selectedLayer%22%3A%7B%22layer%22%3A%22jpeg%22%2C%22visible%22%3Atrue%7D%7D"]
        except Exception as e:
            return [str(e)]

    @classmethod
    def from_path(cls, model_dir):
        """Creates an instance of MyPredictor using the given path.

        This loads artifacts that have been copied from your model directory in
        Cloud Storage. MyPredictor uses them during prediction.

        Args:
            model_dir: The local directory that contains the trained Keras
                model with metadata json
        Returns:
            An instance of `MyPredictor`.
        """
        #tf.enable_eager_execution()

        meta_path = os.path.join(model_dir, 'meta.json')
        model = tf.keras.models.load_model(model_dir, compile=False)
        meta = json.load(open(meta_path))
        
        # hack to load weights
        weights_file = os.path.join(model_dir, 'weights.h5')
        model.save_weights(weights_file, save_format='h5')
        model.load_weights(weights_file)

        return cls(model, meta, model_dir)
