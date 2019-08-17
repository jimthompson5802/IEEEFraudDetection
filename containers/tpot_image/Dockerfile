FROM continuumio/anaconda3:2019.03


# 
# install additional packages
#
RUN apt-get update && apt-get install -y build-essential && \
    pip install mlflow tpot xgboost


WORKDIR /opt/project
ENV MLFLOW_TRACKING_URI /opt/project/tracking