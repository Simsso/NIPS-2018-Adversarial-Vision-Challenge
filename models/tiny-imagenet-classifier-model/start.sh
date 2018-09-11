#!/usr/bin/env bash

echo "Mounting of $1"
# mount bucket
gcsfuse $1 $1/

# start training
echo "Start training of $2"
nohup tensorboard --logdir=$1/$2/tf_logs > /dev/null &
python src/main.py