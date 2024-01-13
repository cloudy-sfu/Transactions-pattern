import pickle
import numpy as np
from sklearn.metrics import r2_score

# %% Load data.
with open('raw/9_feature_rf_stacking_prediction.pkl', 'rb') as f:
    y_test_hat_stacking = pickle.load(f)
with open('raw/4_estimator_lstm_prediction.pkl', 'rb') as f:
    y_test_hat_lstm = pickle.load(f)
with open('raw/1_ts_testing_normalized.pkl', 'rb') as f:
    _, y_test = pickle.load(f)

# %% Averaging
y_test_hat = np.mean([y_test_hat_stacking, y_test_hat_lstm.flatten()], axis=0)
score_test = r2_score(y_test, y_test_hat)
print(score_test)

# %% Export.
with open('raw/10_feature_rf_hybrid_prediction.pkl', 'wb') as f:
    pickle.dump(y_test_hat, f)
