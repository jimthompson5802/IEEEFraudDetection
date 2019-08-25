FROM continuumio/anaconda3:2019.03


# set up git config
RUN git config --global user.email "jimthompson5802@aol.com" \
    && git config --global user.name "Jim Thompson"

# 
# install additional packages
#
RUN pip install mlflow kaggle xgboost

RUN groupadd -g 1000 ec2-user && \
    useradd -u 1000 -g 1000 ec2-user && \
    mkdir -p /home/ec2-user && \
    chown ec2-user:ec2-user /home/ec2-user && \
    ln -s /opt/project/.kaggle /home/ec2-user/.kaggle


WORKDIR /opt/project
ENV MLFLOW_TRACKING_URI /opt/project/tracking
ENV KAGGLE_PROJECT /opt/project