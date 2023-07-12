import pickle
import pandas as pd
from scipy.stats import ttest_rel
import numpy as np
from itertools import combinations

# %% Load data.
with open('raw/7_feature_rf_stacking_r2.pkl', 'rb') as f:
    results = pickle.load(f)
results = pd.DataFrame(results)
compared_columns = ['deep_forest', 'gbdt', 'rf', 'svr', 'xgb', 'final_regressor']

# %% Compare
p_values = []
for i in range(len(compared_columns)):
    col_to_merge = compared_columns.copy()
    col_to_compare = col_to_merge.pop(i)

    for left_value, right_value in combinations(np.unique(results[col_to_compare]), r=2):
        results_merged = pd.merge(results.loc[results[col_to_compare] == left_value, :],
                                  results.loc[results[col_to_compare] == right_value, :],
                                  on=col_to_merge,
                                  suffixes=(f'_{left_value}', f'_{right_value}'))
        t, p = ttest_rel(results_merged[f'validation_r2_{left_value}'], results_merged[f'validation_r2_{right_value}'])
        p_values.append({
            'compared_column': compared_columns[i],
            'left_value': left_value,
            'right_value': right_value,
            't': t,
            'p': p
        })
p_values = pd.DataFrame(p_values)
p_values.to_csv('results/8_stacking_t.csv', index=False)
