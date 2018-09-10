#!/usr/bin/env bash

# mount bucket
gcsfuse $1 $1/

# start training
nohup tensorboard --logdir=tf_logs > /dev/null &
python src/main.py