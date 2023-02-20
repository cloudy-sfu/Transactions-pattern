import pickle
from sklearn.linear_model import LinearRegression

# %% Load data.
with open('raw/2_feature_rf_training_normalized.pkl', 'rb') as f:
    x, y = pickle.load(f)
with open('raw/2_feature_rf_testing_normalized.pkl', 'rb') as f:
    x_test, y_test = pickle.load(f)

# %% Fit on training set.
optimizer = LinearRegression()
optimizer.fit(x, y.ravel())
score = optimizer.score(x, y.ravel())
print(score)

# Predict on testing set.
y_test_hat = optimizer.predict(x_test)
score_test = optimizer.score(x_test, y_test)
print(score_test)

# %% Export.
with open('raw/3_feature_rf_estimator_lr.pkl', 'wb') as f:
    pickle.dump({
        'Estimator': optimizer, 'R^2': score,
    }, f)
with open('raw/3_feature_rf_estimator_lr_prediction.pkl', 'wb') as f:
    pickle.dump(y_test_hat, f)
