"""
H_0: Hybrid model has the same performance as both LSTM and Stacking
H_1: Hybrid model has different performance to LSTM or Stacking
Also inference LSTM > Stacking
"""
import pickle
import pandas as pd
from scipy.stats import ttest_rel
from itertools import combinations
from sklearn.metrics import r2_score, roc_auc_score
import numpy as np

# %% Load data.
y_test_hat = {}
with open('raw/9_feature_rf_stacking_prediction.pkl', 'rb') as f:
    y_test_hat['Stacking'] = pickle.load(f)
with open('raw/4_estimator_lstm_prediction.pkl', 'rb') as f:
    y_test_hat['LSTM'] = pickle.load(f).flatten()
with open('raw/10_feature_rf_hybrid_prediction.pkl', 'rb') as f:
    y_test_hat['Hybrid'] = pickle.load(f)
with open('raw/1_ts_testing_normalized.pkl', 'rb') as f:
    _, y_test = pickle.load(f)
with open('raw/1_y_scaler.pkl', 'rb') as f:
    y_scaler = pickle.load(f)

# %% Constants.
n_boot = 1000

rs = np.random.RandomState(seed=48079)
p_values = []
idx = np.arange(y_test.shape[0])

# %% In regression.
for left, right in combinations(y_test_hat.keys(), r=2):
    err_left = np.zeros(shape=n_boot)
    err_right = np.zeros(shape=n_boot)
    for i in range(n_boot):
        idx_boot = rs.choice(idx, size=y_test.shape[0], replace=True)
        err_left[i] = r2_score(y_test[idx_boot], y_test_hat[left][idx_boot])
        err_right[i] = r2_score(y_test[idx_boot], y_test_hat[right][idx_boot])
    t, p = ttest_rel(err_left, err_right)
    p_values.append({
        'n_boot': n_boot,
        'n_samples': y_test.shape[0],
        'metric': 'r2',
        'left_value': left,
        'right_value': right,
        't': t,
        'p': p,
    })

# %% In classification.
thresholds = np.array([[43.548], [52.4506], [64.6253]])  # same as `13_eval_clf.py`
thresholds = y_scaler.transform(thresholds).flatten()
y_test_clf = y_test >= thresholds
n_classes = thresholds.shape[0]
for left, right in combinations(y_test_hat.keys(), r=2):
    err_left = np.zeros(shape=(n_boot, n_classes))
    err_right = np.zeros(shape=(n_boot, n_classes))
    for i in range(n_boot):
        idx_boot = rs.choice(idx, size=y_test.shape[0], replace=True)
        y_test_hat_left_boot = y_test_hat[left][idx_boot]
        y_test_hat_right_boot = y_test_hat[right][idx_boot]
        for j in range(n_classes):
            err_left[i, j] = roc_auc_score(y_test_clf[idx_boot, j], y_test_hat_left_boot)
            err_right[i, j] = roc_auc_score(y_test_clf[idx_boot, j], y_test_hat_right_boot)
    err_left = np.mean(err_left, axis=1)
    err_right = np.mean(err_right, axis=1)
    t, p = ttest_rel(err_left, err_right)
    p_values.append({
        'n_boot': n_boot,
        'n_samples': y_test.shape[0],
        'metric': 'mean_auc',
        'left_value': left,
        'right_value': right,
        't': t,
        'p': p,
    })

# %% Export.
p_values = pd.DataFrame(p_values)
p_values.to_csv("results/12_testing_significance.csv", index=False)
