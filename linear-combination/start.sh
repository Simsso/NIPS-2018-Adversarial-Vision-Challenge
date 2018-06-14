#!/usr/bin/env bash

echo "Starting project..."
docker build -t linear-combination .
docker rm lc-container
docker run \
    -v /Users/timodenk/Development/nips-2018/linear-combination/model_dir:/opt/app/model_dir \
    -v /Users/timodenk/Development/nips-2018/linear-combination/tf_logs:/opt/app/tf_logs \
    -p 0.0.0.0:6006:6006 \
    -p 8888:8888 \
    --name "lc-container" \
    linear-combination