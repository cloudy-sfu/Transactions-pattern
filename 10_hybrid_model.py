import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score

# %% Load data.
with open('raw/2_ts_training_normalized.pkl', 'rb') as f:
    x_ts, y = pickle.load(f)
with open('raw/2_ts_testing_normalized.pkl', 'rb') as f:
    x_ts_test, y_test = pickle.load(f)
with open('raw/2_feature_rf_training_normalized.pkl', 'rb') as f:
    x_cs, _ = pickle.load(f)
with open('raw/2_feature_rf_testing_normalized.pkl', 'rb') as f:
    x_cs_test, _ = pickle.load(f)

# %% Load models.
with open('raw/7_feature_rf_stacking_r2.pkl', 'rb') as f:
    results = pickle.load(f)
results = pd.DataFrame(results)
b = results.loc[
    (~results['deep_forest']) & (results['gbdt']) & (~results['rf']) & (results['svr']) & (results['xgb']) &
    (results['method'] == 'lr')
].index
assert len(b) == 1
b = b[0]
with open('raw/7_feature_rf_stacking.pkl', 'rb') as f:
    stacking = pickle.load(f)[b]
lstm = tf.keras.models.load_model('raw/4_lstm.h5')

# %% Take the average.
y_hat = np.mean([stacking.predict(x_cs)[:, np.newaxis], lstm.predict(x_ts)], axis=0)
score = r2_score(y, y_hat)

y_test_hat = np.mean([stacking.predict(x_cs_test)[:, np.newaxis], lstm.predict(x_ts_test)], axis=0)
score_test = r2_score(y_test, y_test_hat)

print(score)  # 0.7412034220139881
print(score_test)  # 0.7082195518177964

# %% Export.
with open('raw/10_feature_rf_hybrid_prediction.pkl', 'wb') as f:
    pickle.dump(y_test_hat, f)
