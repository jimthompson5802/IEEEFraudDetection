#!/usr/bin/env bash

set -x -e

container_parm=${1:-pybash}  #pybash | pyjpynb  | h2obash | h2ojpynb | mlflow


case ${container_parm} in
    pybash) image=kag_python
            interactive='-it'
            ports=""
            cmd=/bin/bash;;

    pyjpynb) image=kag_python
             interactive='--detach'
             ports='-p 8888:8888'
             cmd="jupyter notebook --no-browser --ip 0.0.0.0 \
                --allow-root --password='' --NotebookApp.token='' \
                 --notebook-dir=/opt/project";;

    h2obash) image=kag_h2o
            interactive='-it'
            ports=""
            cmd=/bin/bash;;

    h2ojpynb) image=kag_python
             interactive='--detach'
             ports='-p 8888:8888'
             cmd="jupyter notebook --no-browser --ip 0.0.0.0 \
                --allow-root --password='' --NotebookApp.token='' \
                 --notebook-dir=/opt/project";;

    mlflow) image=kag_python
            interactive='--detach'
            ports="-p 5000:5000"
            cmd="mlflow server --host 0.0.0.0 --backend-store-uri=file:///opt/project/tracking";;

esac


docker run ${interactive} --rm \
    -e MFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
    -e INSIDE_DOCKER=true \
    -v ${KAGGLE_PROJECT}:/opt/project \
    ${ports} \
    -p 8787:8787 \
    --name ${container_parm} \
    ${image} \
    ${cmd}