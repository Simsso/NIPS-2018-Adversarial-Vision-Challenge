#!/usr/bin/env bash

sleep 3 && open -a "Google Chrome" http://localhost:6006 &
docker exec lc-container /bin/sh -c "tensorboard --logdir /opt/app/out/tf_logs --port 6006"
#tensorboard --logdir ./out/tf_logs --port 6006
