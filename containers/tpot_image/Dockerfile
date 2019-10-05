FROM continuumio/anaconda3:2019.03

# set up git config
RUN git config --global user.email "jimthompson5802@aol.com" \
    && git config --global user.name "Jim Thompson"

# 
# install additional packages
#
ARG MLFLOW_VERSION
RUN apt-get update && apt-get install -y build-essential && \
    pip install mlflow${MLFLOW_VERSION} kaggle tpot xgboost

RUN groupadd -g 1000 ec2-user && \
    useradd -u 1000 -g 1000 ec2-user && \
    mkdir -p /home/ec2-user && \
    chown ec2-user:ec2-user /home/ec2-user && \
    ln -s /opt/project/.kaggle /home/ec2-user/.kaggle

WORKDIR /opt/project
ENV MLFLOW_TRACKING_URI /opt/project/tracking