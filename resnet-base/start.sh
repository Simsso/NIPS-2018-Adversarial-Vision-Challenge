#!/usr/bin/env bash

cd vq-layer && pip install . && cd ..
python setup.py install && \
python -m resnet_base \
    --name "${MODEL_ID}" \
    --learning_rate 1 \
    --num_epochs 100 \
    --physical_batch_size 32 \
    --virtual_batch_size_factor 8 \
    --pretrained_checkpoint "${HOME}/.data/tiny_imagenet_alp05_2018_06_26.ckpt" \
    --data_dir "${HOME}/.data/tiny-imagenet-200" \
    --save_dir "gs://${BUCKET_NAME}/model_data/${MODEL_ID}/checkpoints" \
    --train_log_dir "gs://${BUCKET_NAME}/model_data/${MODEL_ID}/train" \
    --val_log_dir "gs://${BUCKET_NAME}/model_data/${MODEL_ID}/val" > log_output.txt 2>&1

sleep infinity
