FROM continuumio/anaconda3:2019.03


# 
# install additional packages
#
RUN pip install mlflow xgboost mlbox

WORKDIR /opt/project
ENV MLFLOW_TRACKING_URI /opt/project/tracking