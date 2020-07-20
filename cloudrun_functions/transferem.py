"""Web server that has endpoints to write aligned data to cloud storage.
"""

import os

from flask import Flask, Response, request, make_response, abort
from flask_cors import CORS
import json
import logging
import pwd
import numpy as np
import tensorstore as ts
import traceback
import sys

app = Flask(__name__)

# TODO: Limit origin list here: CORS(app, origins=[...])
CORS(app)
logger = logging.getLogger(__name__)

@app.route('/volume', methods=["POST"])
def volume():
    """Retrieve volume from tensorstore.
    """
    try:
        config_file  = request.get_json()
        
        location = config_file["location"] # contains source and destination
        start = config_file["start"] # contains source and destination
        size = config_file["size"] # contains source and destination
        scale_index = config_file.get("scale_index", 0)
        
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

        data = dataset[start[0]:(start[0]+size[0]), start[1]:(start[1]+size[1]), start[2]:(start[2]+size[2])].read().result()
        r = make_response(data.tobytes())
        r.headers.set('Content-Type', 'application/octet-stream')
        return r
    except Exception as e:
        return Response(traceback.format_exc(), 400)


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
