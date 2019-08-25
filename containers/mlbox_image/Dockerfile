#FROM continuumio/anaconda3:2019.03
FROM continuumio/anaconda3:5.2.0


# set up git config
RUN git config --global user.email "jimthompson5802@aol.com" \
    && git config --global user.name "Jim Thompson"

# 
# install additional packages
#
RUN apt-get update && apt-get install -y build-essential && \
    pip install mlflow mlbox

# upgrade to latest pandas to match rest of environment py3.6
RUN pip install pandas==0.24.2

WORKDIR /opt/project
ENV MLFLOW_TRACKING_URI /opt/project/tracking