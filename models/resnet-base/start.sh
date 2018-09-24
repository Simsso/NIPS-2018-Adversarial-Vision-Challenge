#!/usr/bin/env bash

python setup.py install
python -m resnet_base \
    --pretrained_checkpoint "${HOME}/.data/tiny_imagenet_alp05_2018_06_26.ckpt" \
    --data_dir "${HOME}/.data/tiny-imagenet-200" \
    --train_log_dir "gs://${BUCKET_NAME}/model_data/${MODEL_ID}/train" \
    --val_log_dir "gs://${BUCKET_NAME}/model_data/${MODEL_ID}/val"

sleep infinity
