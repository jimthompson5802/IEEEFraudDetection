#!/usr/bin/env bash

set -x -e

container_parm=${1:-pybash}  #pybash | pyjpynb | mlboxbash | mlboxjpynb | h2obash | h2ojpynb | mlflow
project_dir=${KAGGLE_PROJECT:-$PWD}

case ${container_parm} in
    pybash) image=kag_python
            interactive='-it'
            ports=""
            other_run_parms='--user ec2-user:ec2-user'
            cmd=/bin/bash;;

    pyjpynb) image=kag_python
             interactive='--detach'
             ports='-p 8888:8888'
             other_run_parms='--user ec2-user:ec2-user'
             cmd="jupyter notebook --no-browser --ip 0.0.0.0 \
                --allow-root --password='' --NotebookApp.token='' \
                 --notebook-dir=/opt/project";;

    mlboxbash) image=kag_mlbox
            interactive='-it'
            ports=""
            other_run_parms='--user ec2-user:ec2-user'
            cmd=/bin/bash;;

    mlboxjpynb) image=kag_mlbox
             interactive='--detach'
             ports='-p 8889:8889'
             other_run_parms='--shm-size=1g --user ec2-user:ec2-user'
             cmd="jupyter notebook --no-browser --ip 0.0.0.0  --port 8889\
                --allow-root --password='' --NotebookApp.token='' \
                 --notebook-dir=/opt/project";;

    mlboxgpubash) image=kag_mlboxgpu
            interactive='-it'
            ports=""
            other_run_parms='--user ec2-user:ec2-user'
            cmd=/bin/bash;;


    mlboxgpujpynb) image=kag_mlboxgpu
             interactive='--detach'
             ports='-p 8889:8889'
             other_run_parms='--shm-size=1g --user ec2-user:ec2-user'
             cmd="jupyter notebook --no-browser --ip 0.0.0.0  --port 8889\
                --allow-root --password='' --NotebookApp.token='' \
                 --notebook-dir=/opt/project";;


    h2obash) image=kag_h2o
            interactive='-it'
            ports=""
            other_run_parms='--user ec2-user:ec2-user'
            cmd=/bin/bash;;

    h2ojpynb) image=kag_h2o
             interactive='--detach'
             ports='-p 8890:8890'
             other_run_parms='--user ec2-user:ec2-user'
             cmd="jupyter notebook --no-browser --ip 0.0.0.0 --port 8890\
                --allow-root --password='' --NotebookApp.token='' \
                 --notebook-dir=/opt/project";;

    tpotbash) image=kag_tpot
            interactive='-it'
            ports=""
            other_run_parms='--user ec2-user:ec2-user'
            cmd=/bin/bash;;

    tpotjpynb) image=kag_tpot
             interactive='--detach'
             ports='-p 8891:8891'
             other_run_parms='--user ec2-user:ec2-user'
             cmd="jupyter notebook --no-browser --ip 0.0.0.0 --port 8891\
                --allow-root --password='' --NotebookApp.token='' \
                 --notebook-dir=/opt/project";;

    tfgpubash) image=kag_tfgpu
            interactive='-it'
            ports=""
            other_run_parms='--user ec2-user:ec2-user'
            cmd=/bin/bash;;

    tfgpujpynb) image=kag_tfgpu
             interactive='--detach'
             ports='-p 8892:8892'
             other_run_parms='--user ec2-user:ec2-user'
             cmd="jupyter notebook --no-browser --ip 0.0.0.0 --port 8892\
                --allow-root --password='' --NotebookApp.token='' \
                 --notebook-dir=/opt/project";;

    mlflow) image=kag_python
            interactive='--detach'
            ports="-p 5000:5000"
            other_run_parms='--user ec2-user:ec2-user'
            cmd="mlflow server --host 0.0.0.0 --backend-store-uri=file:///opt/project/tracking";;

esac


docker run ${interactive} --rm \
    -e INSIDE_DOCKER=true \
    -v ${project_dir}:/opt/project \
    ${ports} \
    ${other_run_parms} \
    --hostname ${container_parm} \
    --name ${container_parm} \
    ${image} \
    ${cmd}