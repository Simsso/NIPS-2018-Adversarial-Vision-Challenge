#!/usr/bin/env bash

# mount bucket
gcsfuse --implicit-dirs $BUCKET_NAME $BUCKET_NAME/

# create output folder
mkdir -p $BUCKET_NAME/$MODEL_ID

# start training
python src/main.py