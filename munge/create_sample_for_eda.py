#
# create sample extract as mlflow artifacts
#

import pandas as pd
import numpy as np
import os
import os.path
import tempfile
import shutil
import mlflow
from utils.kaggle import get_global_parameters

#%%
global_parms = get_global_parameters()
DATA_DIR = os.path.join(global_parms['PROJ_DIR'],'data')

#%%
tmpdir = tempfile.mkdtemp()

#%%
# combine train identity and data components
train_data = pd.read_csv(os.path.join(DATA_DIR, 'raw', 'train_transaction.csv'))
train_id = pd.read_csv(os.path.join(DATA_DIR, 'raw', 'train_identity.csv'))
train_raw= pd.merge(train_data,train_id, how='left', on='TransactionID')

#%%
# extract sample for analysis
sample_df = train_raw.sample(frac=0.2, random_state=13)
sample_df.to_pickle(os.path.join(tmpdir,'sample.pkl'))

#%%
# save sample as mlflow artifact
experiment_id = mlflow.set_experiment('feature_set')

#%%
with mlflow.start_run(experiment_id=experiment_id, run_name='eda_sample'):
    mlflow.log_artifact(os.path.join(tmpdir,'sample.pkl'))

#%%
# clean up temp directory
shutil.rmtree(tmpdir)