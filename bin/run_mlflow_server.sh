#!/usr/bin/env bash

# run mlflow server in the docker container

mlflow server --host 0.0.0.0 --backend-store-uri=file:///opt/project/tracking