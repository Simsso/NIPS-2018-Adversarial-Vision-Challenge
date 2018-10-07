#!/usr/bin/env bash

# from https://stackoverflow.com/a/24112741
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

pip install vq-layer

cd resnet-base
python3 setup.py install

echo "Starting model server..."
python3 -m resnet_base \
    --save_dir "$parent_path/weights/"

