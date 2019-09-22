#FROM continuumio/anaconda3:2019.03
FROM continuumio/anaconda3:latest


# set up git config
RUN git config --global user.email "jimthompson5802@aol.com" \
    && git config --global user.name "Jim Thompson"

# 
# install additional packages
#
#RUN apt-get update && apt-get install -y build-essential && \
RUN /opt/conda/bin/pip install mlflow mlbox==0.8.1

# upgrade to latest pandas to match rest of environment py3.6
#RUN pip install pandas==0.24.2 scikit-learn==0.20.4

RUN /opt/conda/bin/pip install kaggle

RUN groupadd -g 1000 ec2-user && \
    useradd -u 1000 -g 1000 ec2-user && \
    mkdir -p /home/ec2-user && \
    chown ec2-user:ec2-user /home/ec2-user && \
    ln -s /opt/project/.kaggle /home/ec2-user/.kaggle
#    mkdir -p /home/ec2-user/.config/matplotlib && \
#    echo "backend: Agg" > /home/ec2-user/.config/matplotlib/matplotlibrc && \
#    chown -R ec2-user:ec2-user /home/ec2-user/.config

USER ec2-user

WORKDIR /opt/project
ENV MLFLOW_TRACKING_URI /opt/project/tracking
ENV KAGGLE_PROJECT /opt/project
ENV PATH /opt/conda/bin:$PATH