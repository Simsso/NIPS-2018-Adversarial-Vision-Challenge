#!/usr/bin/env bash

# add this when the vq-layer library is needed
# cd vq-layer && pip install . && cd ..

cd ..
python setup.py install && \

echo "Starting model server..."
python -m resnet_base \
    --pretrained_checkpoint "${HOME}/.data/tiny_imagenet_alp05_2018_06_26.ckpt"
    # when available, add --global_checkpoint "${HOME}/.data/resnet_base_global.ckpt"
