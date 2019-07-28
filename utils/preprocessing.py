#
#  feature generation functions
#
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def encode_mean_level(cat_x, y):
    """
    cat_x: pandas dataframe for categorical variables
    y: pandas series for response variable.

    Assumes dataframe and series indices are the same

    returns:
      mean-level encoded dataframe
      dictitionary to be used for encoding test predictors

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