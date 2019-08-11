FROM continuumio/anaconda3:5.2.0


# 
# install additional packages
#
RUN pip install mlflow xgboost

WORKDIR /opt/project
ENV MLFLOW_TRACKING_URI /opt/project/tracking