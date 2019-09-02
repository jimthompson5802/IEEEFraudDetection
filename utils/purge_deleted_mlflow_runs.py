import yaml
import shutil
import os.path
import glob
import mlflow

#%%
# retrieve location for mlflow tracking
TRACKING_DIR = mlflow.get_tracking_uri()

#%%
# retrieve experiment directories
experiment_dirs = glob.glob(os.path.join(TRACKING_DIR,'[0-9]'))

#%%
# cycle through run directories looking for deleted runs
print('starting purge run')
for run_dir in [run_dir for d in experiment_dirs for run_dir in glob.glob(os.path.join(d, '[a-f0-9]*'))]:

    # retrieve run meta data file
    with open(os.path.join(run_dir,'meta.yaml'), 'r') as f:
        run_meta_data = yaml.safe_load(f)

    if run_meta_data['lifecycle_stage'] == 'deleted':
        print('purging: ', run_dir)
        shutil.rmtree(run_dir)

print('completed purge run')

