import pickle
import pandas as pd

# %% Load model.
with open('raw/7_feature_rf_stacking_r2.pkl', 'rb') as f:
    results = pickle.load(f)
results = pd.DataFrame(results)
b = results.loc[
    (~results['gbdt']) & (~results['rf']) & (results['svr']) & (results['xgb']) &
    (results['method'] == 'lr')
].index
assert len(b) == 1
b = b[0]
with open('raw/7_feature_rf_stacking.pkl', 'rb') as f:
    optimizer = pickle.load(f)[b]

# %% Load data.
with open('raw/2_feature_rf_training_normalized.pkl', 'rb') as f:
    x, y = pickle.load(f)
with open('raw/2_feature_rf_testing_normalized.pkl', 'rb') as f:
    x_test, y_test = pickle.load(f)

# %% Get training score.
score = optimizer.score(x, y.ravel())
print(score)

# %% Predict on testing set.
y_test_hat = optimizer.predict(x_test)
score_test = optimizer.score(x_test, y_test.ravel())
print(score_test)

# %% Export.
with open('raw/9_feature_rf_stacking.pkl', 'wb') as f:
    pickle.dump(optimizer, f)
with open('raw/9_feature_rf_stacking_prediction.pkl', 'wb') as f:
    pickle.dump(y_test_hat, f)
