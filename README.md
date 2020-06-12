# transfer\_em

This package implements a CycleGAN designed to style transfer one 3D electron microscopy (EM)
dataset to another.  One potential use case would be to transfer data X to Y where
pretrained classifiers already exist for Y.  Another application could be to 'prettify'
data.

The network implements a generator/discriminator network using Tensorflow 2 and works
only with 1 channel, 8 bit data (either 2D or 3D isotropic/near-isotropic).

Features:

* Ability to create tensorflow datasets from 2D or 3D numpy arrays or by fetching dynamically from supported
datasources (DVID and google pre-computed)
* Dataset augmentation and warping mapping routines (mostly useful for debugging)
* Programmable options for specifying 2D or 3D networks and different model sizes
* Example python notebooks for training and running inference
* Helper functions for running inference on large subvolumes
* (hopefully soon) Support for serving a model on Google AI Platform (see below)

Network design:

* Generator: u-net with 3 strided downsample layers with 3x3 convolutions in between, followed by up-sample counterparts
* Discrimiantor: 3 strided downsample layers with 3x3 convolution layers in between
* Discriminator, identity, and cycle loss terms
* Fully convolutional, all VALID convolutions

## Installation and use

To run inference from a pre-trained model (see notebook example at examples/run_local_predict.ipynb),
one just needs to have tensorflow 2.2 installed and matplotlib and then build this package:

    % python setup.py install

To perform training and use some of the other features in this package:

    % pip install tqdm pillow

Install pydot/pydotplus/graphviz to use tensorflow's model architecture plotting.

One can learn the training and inference routines with the included 2D example using
examples/simple_training.ipynb.  There is an example notebook (examples/generator_training3D.ipynb)
for training 3D data
fetched dynamically from the cloud.
To run this example and access the cloud data, one needs to create a cloud run http service (see cloudrun_functions/README.md).
The data is stored [neuroglancer precomputed](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed) format.
(Note: in principle, the fetching of precomputed data does not
require a separate HTTP service, but making a separate web service greatly simplified the software
depenedencies for transfer_em.  This situation might change in the future.)
Alternatively, one can pre-download the data from the cloud and create a dataset as is
done for the 2D example.

## Google AI Platform

Google Cloud provides a set of managed tools for training and using deep networks.

Conveniently, Google provides VM images with pre-installed deep learning libraries (documented
[here](https://cloud.google.com/ai-platform/deep-learning-vm/docs)).  One can run the example notebooks in this package using the managed [AI Platform Notebooks](https://cloud.google.com/ai-platform-notebooks).    The managed platform above makes
launching notebooks or running headless training scripts straightforward (transfer_em still needs to be installed on this
VM as documented above but tensorflow is already installed and configured).  Note:
the notebook application should be shutdown to release the GPU if one is running scripts via the command-line on the VM. 

### Hosting a model in the cloud

One of the nice features of Google AI Platform is the ability to save a model
to the cloud and run inference using an auto-scaled serverless endpoint ([here](https://cloud.google.com/ai-platform/prediction/docs/deploying-models#console)).  This enables inference to be run via
a simple HTTP or Python API.  transfer_em requires some customized pre and post-processing steps
that should ideally be run on the remote platform.  Unfortunately, as of this time, Tensorflow >2 is not supported
with customized prediction routines.

When support for Tensorflow >2 is provided, one can perform the following steps
to host the model and perform inference (this workflow is untested):

* save the model based on a training checkpoint using the script bin/saved_model.py
* copy the saved model directory to a Google storage bucket
* upload a python build of transfer_em (see command below) to a Google storage bucket (it can be the same bucket as the saved model)

      % python setup.py sdist --formats=gztar 

* load the model into the AI platform [here](https://console.cloud.google.com/ai-platform/models)
* create a version of that model following the UI prompts (options: python 3.7, Custom Prediction Routines,
select the directory of the saved model, select the transfer_em package tar file, set the prediction class to predict_service.TransferEMPredictor)

transfer_em provides a 'predict_service' that implements a custom python routine for inference
on the AI platform.  To use this service, see the examples/run_cloud_predict_service.ipynb.  (Note:
currently this service is setup to use CloudRun to access the precomputed image volumes.)

## TODO

* Allow users to provide a pre-trained classifier (for a desired target application) to be
used to constrain the transfer_em discriminator
* Provide an option for simplifying the network for faster training or debugging
* Implement mirrored training for multiple GPUs (this requires some of the loss
calculations to be modified slightly)
* Test and deploy cloud-based model hosting and inference on Google AI Platform
* Use pre-built tensorstore installation to avoid needing Google Cloud Run for pre-computed cloud data fetchiing
