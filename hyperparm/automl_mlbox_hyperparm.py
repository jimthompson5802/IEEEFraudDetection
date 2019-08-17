
# coding: utf-8

# # mlbox do some hyper param analysis

# In[1]:


import pandas as pd
import numpy as np
import os
import os.path
import sys
import tempfile
import shutil

import mlflow
import tempfile


# In[2]:


from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *


# In[3]:


sys.path.append('..')
from utils.kaggle import get_global_parameters
from utils.mlflow_experiments import retrieve_artifacts, extract_run_data_for_experiment
global_parms = get_global_parameters()


# In[4]:


global_parms


# In[5]:


TMPDIR= tempfile.mkdtemp()


# In[6]:


MLBOX_SAVE='mlbox_save'


# ## Retrieve training data

# In[7]:


# retrieve run_id for desired feature set to test
run_info = extract_run_data_for_experiment('feature_set')
RUN_ID = run_info.loc[run_info['mlflow.runName'] == 'feature_set_0'].run_id.values[0]

retrieve_artifacts(RUN_ID, '.', TMPDIR)


# In[8]:


os.listdir(TMPDIR)


# In[9]:


pd.read_pickle(os.path.join(TMPDIR,'fs_train_df.pkl'))     .sample(10000).drop(['TransactionID', 'TransactionDT', 'addr1'], axis=1)     .to_csv(os.path.join(TMPDIR,'fs_train_df.csv'),index=False)


# In[10]:


pd.read_pickle(os.path.join(TMPDIR,'fs_test_df.pkl'))     .sample(10000).drop(['isFraud', 'TransactionID', 'TransactionDT', 'addr1'], axis=1)     .to_csv(os.path.join(TMPDIR,'fs_test_df.csv'),index=False)


# In[11]:


paths = [os.path.join(TMPDIR,'fs_test_df.csv'), os.path.join(TMPDIR,'fs_train_df.csv')]


# In[12]:


rd = Reader(sep=',', to_path=MLBOX_SAVE)
df = rd.train_test_split(paths, 'isFraud')


# In[13]:


dft = Drift_thresholder(to_path=MLBOX_SAVE)
df = dft.fit_transform(df)


# In[14]:


opt = Optimiser(scoring = 'roc_auc', n_folds = 3, to_path=MLBOX_SAVE)


# In[15]:


opt.evaluate(None, df)


# In[16]:


space = {
    
        'ne__numerical_strategy':{"search":"choice",
                                 "space":[0]},
        'ce__strategy':{"search":"choice",
                        "space":["label_encoding","random_projection", "entity_embedding"]}, 
        'fs__threshold':{"search":"uniform",
                        "space":[0.01,0.3]},    
        'est__max_depth':{"search":"choice",
                                  "space":[3,4,5,6,7]},
        'est__n_estimators': {'search':'choice', 'space':[250, 500, 750, 1000, 1500]}
    
        }

best = opt.optimise(space, df)


# In[17]:


best


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
prd = Predictor(to_path=MLBOX_SAVE)
prd.fit_predict(best, df)


# In[20]:


get_ipython().run_cell_magic('javascript', '', "IPython.notebook.kernel.execute(`notebookName = '${IPython.notebook.notebook_name}'`);")


# In[22]:


notebookName


# In[25]:


# remove joblib artifacts from MLBOX_SAVE, these are not needed
shutil.rmtree(os.path.join(MLBOX_SAVE,'joblib'))


# In[26]:


# save sample as mlflow artifact
experiment_id = mlflow.set_experiment('hyperparms')

#%%
with mlflow.start_run(experiment_id=experiment_id, run_name='mlbox_inital_hyperparm'):
    mlflow.log_param('notebook_name',notebookName)
    mlflow.log_artifacts(MLBOX_SAVE)


# ## Clean-up

# In[17]:


shutil.rmtree(TMPDIR)


# In[ ]:


pd.__version__

