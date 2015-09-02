#!/bin/bash

virtualenv ./_env
source ./_env/bin/activate

pip install https://github.com/inducer/loopy/tarball/master nose numpy pyopencl
python setup.py develop
