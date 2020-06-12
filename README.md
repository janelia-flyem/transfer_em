# transfer\_em

This package implements a CycleGANs designed to style transfer one 3D electron microscopy
dataset (EM) to another.  One potential use case would be to transfer data X to Y where
pretrained classifiers already exist for Y.  Another application could be to 'prettify'
data.

The network implements a generator/discriminator network using Tensorflow 2 and works
only with 1 channel, 8 bit data (either 2D or 3D isotropic/near-isotropic).

Features:

* Ability to specify datasets with 2D or 3D numpy arrays or using supported
datasources
* Augmentation and warping dataset mapping options (mostly useful for debugging)
* Options for specifying 2D or 3D network and different model sizes
* Example python notebooks for training and running inference
* Helper functions for running inference on large subvolumes
* (hopefully soon) Support for serving model on Google AI Platform (see below)

Network design:

* Generator: u-net with 3 strided downsample layers, and 3x3 convolutions in between
* Discrimiantor: 3 strided downsample with 3x3 convolution layers in between
* Discriminator, identity, and cycle loss terms
* Fully convolutional, all VALID convolutions

# Installation and use

To run inference from a pre-trained model (see notebook example at examples/run_local_predict.ipynb),
one just needs to have tensorflow 2.2 installed and then build this package:

% python setup.py install

To perform training and use some of the other features in this package:

% pip install tqdm pillow

Install pydot/pydotplus/graphviz to use tensorflow's model output.

One can learn the training and inference routines with an included 2D example using
examples/simple_training.ipynb.  There is an example of fetching data dynamically
from [neuroglancer precomputed](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed) formatted data stored in the cloud here: example/generator_training3D.ipynb.
To do this, one needs to create a service to access this data (see cloudrun_functions/README.md).
Alternatively, one can pre-download the data from the cloud and create a dataset as is
done for the 2D example.  (Note: in principle, the fetching of precomputed data does not
require a separate service but making a separate web service greatly simplified the software
depenedencies for this package.  This situation might change in future versions.)


# Google AI Platform

TBD


# TODO

* Allow users to provide a pre-trained classifier (for a target application) to be
used to constrain the discriminator
* Provide option for simplifying the network to enable a smaller model

