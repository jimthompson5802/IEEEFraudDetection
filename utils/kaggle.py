#
#  Collection of Kaggle-related functions
#

import os
from sklearn.metrics import roc_auc_score



# global parameters
def get_global_parameters():
    inside_docker = os.getenv('INSIDE_DOCKER')

    if inside_docker is not None:
        global_parms = {
            'PROJ_DIR': '/opt/project'
        }
    else:
        global_parms = {
            'PROJ_DIR': os.getcwd()
        }

    return global_parms



# return metric for the contest
def calc_contest_metric(y_true, y_score):
    return roc_auc_score(y_true,y_score)