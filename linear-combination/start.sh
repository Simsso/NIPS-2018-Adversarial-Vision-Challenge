#!/usr/bin/env bash

echo "Starting project..."
docker build -t linear-combination .
docker rm lc-container
# mount TensorBoard folder as volume: -v tf_logs:/opt/app/tf_logs 
docker run -p 0.0.0.0:6006:6006 -p 8888:8888 --name "lc-container" linear-combination