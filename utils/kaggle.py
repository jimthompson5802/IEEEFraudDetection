#
#  Collection of Kaggle-related functions
#

from sklearn.metrics import roc_auc_score

# return metric for the contest
def calc_contest_metric(y_true, y_score):
    return roc_auc_score(y_true,y_score)