#
# Feature Set 0
# All numeric and categorical features that are deemed signification from
# chi-square analysis.
# Keep raw values
#

import pandas as pd
import numpy as np
import os
import os.path
import mlflow

from sklearn.model_selection import train_test_split

from utils.preprocessing import retrieve_predictors
from utils.kaggle import get_global_parameters

import tempfile
import pickle
import shutil


#%%
global_parms = get_global_parameters()
DATA_DIR = os.path.join(global_parms['PROJ_DIR'],'data')

#%%
#
# Retrieve eda results to get significant categorical and numeric attributes
#
RUN_ID_NUM='9b062ab25fdb46db8f984bf674381432'  # run id for numeric predictors eda
RUN_ID_CAT='d6a672688da343e089f17714f6ae494b'  # run id for categorical predictors eda

cat_predictors, num_predictors_skewed, num_predictors_nonskewed = retrieve_predictors(RUN_ID_NUM, RUN_ID_CAT)

#%%
#
# retrieve Kaggle raw data
#
train_raw = pd.read_pickle(os.path.join(DATA_DIR, 'raw', 'train_combined.pkl'))
test_raw = pd.read_pickle(os.path.join(DATA_DIR,'raw','test_combined.pkl'))

#%%
# start building up feature set dataframe
train_raw_df = pd.concat([train_raw[['isFraud', 'TransactionID', 'TransactionDT']],
                   train_raw[cat_predictors],
                   train_raw[num_predictors_skewed],
                   train_raw[num_predictors_nonskewed]
                   ], axis=1).copy()

#%%
test_raw_df = pd.concat([test_raw[['TransactionID', 'TransactionDT']],
                   test_raw[cat_predictors],
                   test_raw[num_predictors_skewed],
                   test_raw[num_predictors_nonskewed]
                   ], axis=1).copy()

#%%
# split train, valid, test data sets
fs_train_df, temp_train  = train_test_split(train_raw_df, train_size=0.8, random_state=13, stratify=train_raw_df['isFraud'])
fs_valid_df, fs_test_df = train_test_split(temp_train, train_size=0.5, random_state=29, stratify=temp_train['isFraud'])

kag_test_df = test_raw_df

#%%
fs_list = ['fs_train_df', 'fs_valid_df', 'fs_test_df', 'kag_test_df']
tmpdir = tempfile.mkdtemp()
for f in fs_list:
    eval(f).to_pickle(os.path.join(tmpdir, f + '.pkl'))

#%%
# Save feature set as mlflow artifacts
experiment_id = mlflow.set_experiment('feature_set')
with mlflow.start_run(experiment_id=experiment_id, run_name='feature_set_0'):
        mlflow.log_artifacts(tmpdir)

#%%
# clean up
shutil.rmtree(tmpdir)
