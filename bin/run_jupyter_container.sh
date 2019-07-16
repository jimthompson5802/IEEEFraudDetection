#!/usr/bin/env bash

docker run -it --rm \
    -e MFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
    -v ${KAGGLE_PROJECT}:/opt/project \
    -p 5000:5000 \
    -p 8888:8888 \
    --name kag_jupyter \
    kag_python \
    jupyter notebook --no-browser --ip 0.0.0.0 \
          --allow-root --password='' --NotebookApp.token='' \
          --notebook-dir=/opt/project