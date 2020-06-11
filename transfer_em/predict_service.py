"""Function for handling Google AI Platform prediction calls.

Based on example from https://cloud.google.com/ai-platform/prediction/docs/custom-prediction-routines.
"""

import os
import json

import numpy as np
import tensorflow as tf
from .utils import predict_ng_cube
import base64

class TransferEMPredictor(object):
    """Prediction utility from tranfering EM to another EM domain.
    """

    def __init__(self, model, meta):
        """Stores artifacts for prediction. Only initialized via `from_path`.
        """
        self._model = model
        self.outdimsize = meta["outdimsize"] 
        self.buffer = meta["buffer"]
        self.meanstd_x = meta["meanstd_x"]
        self.meanstd_y = meta["meanstd_y"]

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
        if len(instances) != 1:
            raise RuntimeError("only one instance allowed")
    
        location = instances[0]["location"]
        cloudrun = instances[0]["cloudrun"]
        start = tuple(instances[0]["start"])
        size = tuple(instances[0]["size"])

        # run prediction
        res = predict_ng_cube(location, start, size, self._model, self.meanstd_x, self.meanstd_y, cloudrun, outdimsize=self.outdimsize, buffer=self.buffer) 
        res_hex = base64.b64encode(res)
        return [res_hex]  

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
        meta_path = os.path.join(model_dir, 'meta.json')
        model = tf.keras.models.load_model(model_dir, compile=False)
        meta = json.load(open(meta_path))
        
        return cls(model, meta)
