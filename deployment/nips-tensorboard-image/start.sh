#!/usr/bin/env bash

# mount bucket
gcsfuse $BUCKET_NAME $BUCKET_NAME/

# start training
nohup tensorboard --port=80 --logdir=$BUCKET_NAME/ > /dev/null & sleep infinity