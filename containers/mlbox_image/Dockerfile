FROM continuumio/anaconda3:5.2.0

# 
# install additional packages
#
RUN pip install mlflow mlbox

# upgrade to latest pandas to match rest of environment
RUN pip install pandas==0.24.2

WORKDIR /opt/project
ENV MLFLOW_TRACKING_URI /opt/project/tracking