#!/usr/bin/env bash

# run mlflow server in the docker container

docker run --detach --rm \
    -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
    -v ${KAGGLE_PROJECT}:/opt/project \
    -p 5000:5000 \
    --name mlflow_server \
    kag_python \
    mlflow server --host 0.0.0.0 --backend-store-uri=file:///opt/project/tracking