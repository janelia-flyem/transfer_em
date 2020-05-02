#! /usr/bin/env python
from setuptools import setup, find_packages

with open('./README.md') as f:
    long_description = f.read()

setup(name='transfer_em',
      version='0.1',
      author='Stephen Plaza',
      description='CycleGans for EM transfer learning',
      long_description=long_description,
      author_email='plazas@janelia.hhmi.org',
      url='https://github.com/janelia-flyem/transfer_em',
      packages=find_packages(),
      )

