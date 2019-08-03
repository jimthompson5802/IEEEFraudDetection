#!/usr/bin/env bash

docker run --detach --rm \
    -e MFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
    -e INSIDE_DOCKER=true \
    -v ${KAGGLE_PROJECT}:/opt/project \
    -p 8888:8888 \
    -p 8787:8787 \
    --name kag_jupyter \
    kag_python \
    jupyter notebook --no-browser --ip 0.0.0.0 \
          --allow-root --password='' --NotebookApp.token='' \
          --notebook-dir=/opt/project