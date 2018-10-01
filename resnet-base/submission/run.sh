#!/usr/bin/env bash

# add this when the vq-layer library is needed
# cd vq-layer && pip install . && cd ..

cd ..
python3 setup.py install

echo "Starting model server..."
python3 -m resnet_base \
    --pretrained_checkpoint "${HOME}/.models/tiny_imagenet_alp05_2018_06_26.ckpt"
    # when available, add --global_checkpoint "${HOME}/.models/resnet_base_global.ckpt"
