#!/usr/bin/env bash

# mount bucket
gcsfuse --implicit-dirs $BUCKET_NAME $BUCKET_NAME/

# create output folder
mkdir -p $BUCKET_NAME/$MODEL_ID

# start training
nohup tensorboard --logdir=$BUCKET_NAME/$MODEL_ID/tf_logs > /dev/null &
python src/main.py