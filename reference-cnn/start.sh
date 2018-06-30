#!/usr/bin/env bash

echo "Starting project..."
docker build -t reference-cnn .
docker rm ref-cnn-container
docker run \
    -v /Users/fp/Documents/Software-Dev/Machine_Learning/NIPS-2018/reference-cnn/out:/opt/app/out \
    -p 0.0.0.0:6006:6006 \
    -p 8888:8888 \
    --name "ref-cnn-container" \
    reference-cnn