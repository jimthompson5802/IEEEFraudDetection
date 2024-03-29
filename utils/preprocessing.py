#
#  feature generation functions
#
import pandas as pd
import numpy as np
import os
import os.path
import tempfile

import shutil

from utils.mlflow_experiments import retrieve_artifacts
from utils.kaggle import get_global_parameters

from sklearn.model_selection import StratifiedKFold
from sklearn.base import  BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import FeatureUnion, Pipeline

from io import StringIO



def encode_mean_level(cat_x, y):
    """
    cat_x: pandas dataframe for categorical variables
    y: pandas series for response variable.

    Assumes dataframe and series indices are the same

    returns:
      mean-level encoded dataframe
      dictionary to be used for encoding test predictors

    """

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    results = []

    overall_results = []

    # compute mean-level for training from out-of-sample
    for out_sample_idx, in_sample_idx in skf.split(cat_x, y):
        cat_ml_in = pd.DataFrame()
        for c in cat_x.columns:
            fold_df = pd.DataFrame(pd.concat([cat_x[c].iloc[out_sample_idx], y.iloc[out_sample_idx]], axis=1))
            out_sample_mean_levels = fold_df.groupby(c)['isFraud'].mean()
            cat_ml_in[c + '_ml'] = cat_x[c].iloc[in_sample_idx].map(out_sample_mean_levels.to_dict())
            overall_results.append((c, out_sample_mean_levels))

        results.append(cat_ml_in)

    df9 = pd.DataFrame(overall_results)
    df9.columns = ['var', 'mean_levels']

    def combine_mean_levels(x):
        ll1 = x.to_list()
        return pd.concat(ll1, axis=1).mean(axis=1)

    ll = df9.groupby('var')['mean_levels'].apply(combine_mean_levels)

    test_mean_level_mapping = {}
    for var in ll.index.get_level_values(0):
        test_mean_level_mapping.update({var: ll[var].to_dict()})

    return pd.concat(results, axis=0), test_mean_level_mapping


def retrieve_predictors(num_run_id, cat_run_id):
    """
    Retrieve mlflow artifacts and extract out candidate predictors

    :param num_run_id: mlflow run id where numeric predictors were analyzed
    :param cat_run_id: mlflow run id where categorical predictors were analyzed

    :return: tuple cat_predictors, num_predictors_skewed, num_predictors_nonskewed
    """

    # %%
    global_parms = get_global_parameters()
    DATA_DIR = os.path.join(global_parms['PROJ_DIR'], 'data')

    # %%
    #
    # Retrieve eda results to get significant categorical and numeric attributes
    #
    RUN_ID_NUM = num_run_id  # run id for numeric predictors eda
    RUN_ID_CAT = cat_run_id  # run id for categorical predictors eda

    in_tmpdir = tempfile.mkdtemp()
    num_dir = os.path.join(in_tmpdir, 'num')
    cat_dir = os.path.join(in_tmpdir, 'cat')

    os.mkdir(num_dir)
    os.mkdir(cat_dir)

    retrieve_artifacts(RUN_ID_NUM, '.', num_dir)
    retrieve_artifacts(RUN_ID_CAT, '.', cat_dir)

    # %%
    os.listdir(num_dir)
    # %%
    os.listdir(cat_dir)

    # %%
    # retrieve list of categorical variables
    cat_df = pd.read_pickle(os.path.join(cat_dir, 'chisq_df.pkl'))
    cat_df = cat_df.loc[cat_df.p_value < 0.05 / cat_df.shape[0]]
    cat_predictors = cat_df.index.to_list()

    # retrieve skewed numeric variables
    skewed_df = pd.read_pickle(os.path.join(num_dir,'chisq_df_skewed.pkl'))
    skewed_df = skewed_df.loc[skewed_df.p_value < 0.05/skewed_df.shape[0]]
    num_predictors_skewed = skewed_df['var'].to_list()

    # retrieve nonskewed numeric variables
    nonskewed_df = pd.read_pickle(os.path.join(num_dir,'chisq_df_nonskewed.pkl'))
    nonskewed_df = nonskewed_df.loc[nonskewed_df.p_value < 0.05/nonskewed_df.shape[0]]
    num_predictors_nonskewed = nonskewed_df['var'].to_list()



    shutil.rmtree(cat_dir)
    shutil.rmtree(num_dir)

    return cat_predictors, num_predictors_skewed, num_predictors_nonskewed


class Log1PTransformer(BaseEstimator, TransformerMixin):
    """
    Perform log(1 + X) transformation.
    If any attribute of X contains negative values, that attribute
    will be translated to be non-negative.
    """
    def __init__(self):
        """

        :param predictors: list of predictors to transform
        :return: None
        """

    def fit(self, X):
        """
        :param x: Pandas dataframe containing selected predictors
        :param y: Pandas series on response variable
        :return:
        """

        self.adjustments = np.where(np.min(X) >= 0, 0, -np.min(X))

    def transform(self, X):
        return np.log1p(X + self.adjustments)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        ans = np.exp(X) - 1.0
        ans = ans - self.adjustments
        return ans


class SkewedNumberTransformer(BaseEstimator, TransformerMixin):
    """
    Perform log(1 + X) transformation.
    Augment with indicator matrix for any missing values.
    Imput median for any missing values
    """
    def __init__(self, imputer_strategy='median'):
        """
        :impute_strategy:  strategy to for missing value imputation
        :return: None
        """
        pipe = Pipeline([('log1p', Log1PTransformer()),
                         ('imp_nan', SimpleImputer(strategy=imputer_strategy))])

        self.union = FeatureUnion([('log1p', pipe),
                                   ('nan_ind',MissingIndicator(features='all'))])

    def fit(self, X):
        """
        :param x: Pandas dataframe containing selected predictors
        :param y: Pandas series on response variable
        :return:
        """
        self.union.fit(X,None)

    def transform(self, X):
        df = self.union.transform(X)
        return df

    def fit_transform(self, X):
        self.union.fit(X)
        return self.union.transform(X)

    def inverse_transform(self, X):
        pass


if __name__ == '__main__':
    print('Hello')
    df = pd.read_csv(StringIO("""
x1,x2,x3,x4,x5
0,3.,-2,2,5
1.,4,-5.,,-1
2,5,-7,4.,1.
3,7,,6,3.    
    """))
    print(df)
    print(df.dtypes)

    df0 = pd.read_csv(StringIO("""
x1,x2,x3,x4,x5
0,3.,-2,2,5
1.,,-5.,6,
2,5,-7,4.,1.
3,7,3,6,3.    
        """))
    print(df0)


    log1p_estimator = Log1PTransformer()

    log1p_estimator.fit(df)
    print('log1p_transform\n', log1p_estimator.transform(df))

    log1p_estimator2 = Log1PTransformer()
    print('log1p_fit_transform\n', log1p_estimator.fit_transform(df))

    print("starting skew test suite")
    skewed_transformer = SkewedNumberTransformer()

    skewed_transformer.fit(df)
    df2 = pd.DataFrame(skewed_transformer.transform(df))
    print('df2\n', df2)

    skewed_transformer2 = SkewedNumberTransformer()
    df3 = pd.DataFrame(skewed_transformer2.fit_transform(df))
    df3.index = df.index
    df3.columns = df.columns.to_list() + [c+"_nan" for c in df.columns]
    print('df3\n', df3)

    df4 = pd.DataFrame(skewed_transformer2.transform(df0))
    df4.index = df0.index
    df4.columns = df0.columns.to_list() + [c+"_nan" for c in df0.columns]
    print('df4\n', df4)
