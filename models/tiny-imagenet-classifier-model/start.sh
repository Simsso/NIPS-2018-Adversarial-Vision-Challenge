#!/usr/bin/env bash

nohup tensorboard --logdir=tf_logs > /dev/null &
python src/main.py