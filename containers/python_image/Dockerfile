FROM continuumio/anaconda3:2019.03


# 
# install mlflow and xgboost
#
RUN pip install mlflow xgboost

WORKDIR /opt/project
ENV MLFLOW_TRACKING_URI /opt/project/tracking