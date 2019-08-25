#!/usr/bin/env bash

# invoke kaggle api to retrieve competition data set

if [[ -d /opt/project/data/raw ]]; then
    mkdir -p /opt/project/data/raw
fi

kaggle competitions download -p /opt/project/data/raw ieee-fraud-detection