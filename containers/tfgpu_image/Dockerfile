# Tensorflow
FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get -y update && apt-get install -y git

RUN pip install jupyter matplotlib

ARG MLFLOW_VERSION
RUN pip install mlflow${MLFLOW_VERSION}

RUN groupadd -g 1000 ec2-user && \
    useradd -u 1000 -g 1000 ec2-user && \
    mkdir -p /home/ec2-user && \
    chown ec2-user:ec2-user /home/ec2-user && \
    ln -s /opt/project/.kaggle /home/ec2-user/.kaggle


WORKDIR /opt/project
ENV MLFLOW_TRACKING_URI /opt/project/tracking
ENV KAGGLE_PROJECT /opt/project

