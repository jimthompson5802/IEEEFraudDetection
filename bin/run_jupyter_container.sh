#!/usr/bin/env bash

docker run --detach --rm \
    -e MFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
    -v ${KAGGLE_PROJECT}:/opt/project \
    -p 8888:8888 \
    --name kag_jupyter \
    kag_python \
    jupyter notebook --no-browser --ip 0.0.0.0 \
          --allow-root --password='' --NotebookApp.token='' \
          --notebook-dir=/opt/project