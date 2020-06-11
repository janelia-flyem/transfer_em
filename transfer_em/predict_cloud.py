import json
import googleapiclient.discovery
import numpy as np
import base64

def predict_cloud(project, model, location, cloudrun, start, size, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        location (str): location of source data
        start (tuple(int)): starting x,y,z
        size (tuple(int)): size x,y,z
        version: str, version of the model to target.
    Returns:
        uint8 2D or 3D numpy array
    """
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    service = googleapiclient.discovery.build('ml', 'v1')
    name = f"projects/{project}/models/{model}"

    if version is not None:
        name += f"/versions/{version}"

    # create request dictionary
    payload = [{
        "location": location,
        "cloudrun": cloudrun,
        "start": start,
        "size": size
    }]

    response = service.projects().predict(
        name=name,
        body={'instances': payload}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    # convert to numpy array from base64
    data_str = response['predictions']
    array = np.frombuffer(base64.decodebytes(data_str), dtype=np.uint8)
    return array.reshape(tuple(reversed(size)))
