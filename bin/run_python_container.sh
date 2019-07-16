#!/usr/bin/env bash

docker run -it --rm \
    -e MFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
    -v ${KAGGLE_PROJECT}:/opt/project \
    -p 5000:5000 \
    --name kag_python \
    kag_python \
    /bin/bash