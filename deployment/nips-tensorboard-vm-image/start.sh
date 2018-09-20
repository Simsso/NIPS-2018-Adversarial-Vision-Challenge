#!/usr/bin/env bash

# start tensorboard
while true
do
nohup tensorboard --logdir=gs://nips-2018-adversarial-vision-challenge-data/model_data/ &> /dev/null &
sleep 300;
pkill -f "tensorboard --logdir"
echo "Restarting Tensorboard .."
done