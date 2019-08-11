FROM continuumio/anaconda3:5.2.0


#
# Add h2o python libraries
#
# install based on http://h2o-release.s3.amazonaws.com/h2o/rel-xia/1/docs-website/h2o-docs/downloading.html#install-in-python
#
RUN apt-get -y update && apt-get -y install openjdk-8-jdk &&  \
	pip install requests &&\
	pip install tabulate &&\
	pip install "colorama>=0.3.8" && \
	pip install future && \
    pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o


# 
# install additional packages
#
RUN pip install mlflow

WORKDIR /opt/project
ENV MLFLOW_TRACKING_URI /opt/project/tracking