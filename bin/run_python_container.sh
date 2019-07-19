#!/usr/bin/env bash

docker run -it --rm \
    -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
    -v ${KAGGLE_PROJECT}:/opt/project \
    --name kag_python \
    kag_python \
    /bin/bash