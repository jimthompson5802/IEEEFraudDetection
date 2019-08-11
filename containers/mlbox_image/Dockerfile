FROM continuumio/anaconda3:5.2.0

# 
# install additional packages
#
RUN pip install mlflow mlbox

WORKDIR /opt/project
ENV MLFLOW_TRACKING_URI /opt/project/tracking