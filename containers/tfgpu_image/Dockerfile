# Tensorflow
FROM tensorflow/tensorflow:latest-gpu-py3

RUN pip install jupyter

RUN groupadd -g 1000 ec2-user && \
    useradd -u 1000 -g 1000 ec2-user && \
    mkdir -p /home/ec2-user && \
    chown ec2-user:ec2-user /home/ec2-user && \
    ln -s /opt/project/.kaggle /home/ec2-user/.kaggle


WORKDIR /opt/project
ENV MLFLOW_TRACKING_URI /opt/project/tracking
ENV KAGGLE_PROJECT /opt/project

