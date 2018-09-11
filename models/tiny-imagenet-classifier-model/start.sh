#!/usr/bin/env bash

echo "Mounting of $BUCKET_NAME"
# mount bucket
gcsfuse $BUCKET_NAME $MODEL_ID/

# create output folder
mkdir -p $BUCKET_NAME/$MODEL_ID

# start training
echo "Start training of $LOG_FOLDER"
nohup tensorboard --logdir=$BUCKET_NAME/$MODEL_ID/tf_logs > /dev/null &
python src/main.py