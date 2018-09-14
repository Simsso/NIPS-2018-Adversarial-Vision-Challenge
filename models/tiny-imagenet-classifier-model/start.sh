#!/usr/bin/env bash

# mount bucket
gcsfuse --implicit-dirs $BUCKET_NAME $BUCKET_NAME/

# start training
python src/main.py