FROM continuumio/anaconda3:2019.03

# set up git config
RUN git config --global user.email "jimthompson5802@aol.com" \
    && git config --global user.name "Jim Thompson"

# 
# install additional packages
#
RUN apt-get update && apt-get install -y build-essential && \
    pip install mlflow tpot xgboost


WORKDIR /opt/project
ENV MLFLOW_TRACKING_URI /opt/project/tracking