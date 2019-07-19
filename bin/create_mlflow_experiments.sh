#!/usr/bin/env bash

# create standard experiments

mlflow experiments create  -n feature_set
mlflow experiments create  -n eda
mlflow experiments create  -n hyperparms
mlflow experiments create  -n models
