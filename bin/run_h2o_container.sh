#!/usr/bin/env bash

docker run -it --rm \
    -e MLFLOW_TRACKING_URI=file:///opt/project/tracking \
    -e INSIDE_DOCKER=true \
    -v ${KAGGLE_PROJECT}:/opt/project \
    --name kag_h2o \
    kag_h2o \
    /bin/bash