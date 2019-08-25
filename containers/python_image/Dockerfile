FROM continuumio/anaconda3:2019.03


# set up git config
RUN git config --global user.email "jimthompson5802@aol.com" \
    && git config --global user.name "Jim Thompson"

# 
# install additional packages
#
RUN pip install mlflow xgboost


WORKDIR /opt/project
ENV MLFLOW_TRACKING_URI /opt/project/tracking