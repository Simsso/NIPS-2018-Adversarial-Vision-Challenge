#!/usr/bin/env bash

# start tensorboard
while true
do
nohup tensorboard --logdir=gs://nips-2018-data/model_data/ &> /dev/null &
sleep 600;
pkill -f "tensorboard --logdir"
echo "Restarting Tensorboard .."
done