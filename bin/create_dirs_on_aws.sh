#!/bin/bash
#
# create untracked but required directories on AWS
#

if [[ ! -d ${KAGGLE_PROJECT}/data ]]
then
	mkdir ${KAGGLE_PROJECT}/data
fi


if [[ ! -d ${KAGGLE_PROJECT}/tracking ]]
then
	mkdir ${KAGGLE_PROJECT}/tracking
fi
