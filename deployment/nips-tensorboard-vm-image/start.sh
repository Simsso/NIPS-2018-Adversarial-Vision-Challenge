#!/usr/bin/env bash

# start tensorboard
nohup tensorboard --logdir=gs://nips-2018-adversarial-vision-challenge-data/model_data/ > /dev/null & sleep infinity