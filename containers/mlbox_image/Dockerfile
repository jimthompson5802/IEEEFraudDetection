FROM continuumio/miniconda3:4.3.27

# 
# install additional packages
#
RUN pip install mlflow mlbox

WORKDIR /opt/project
ENV MLFLOW_TRACKING_URI /opt/project/tracking