import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from skopt.space import Real

# %% Load data.
with open('raw/2_feature_rf_training_normalized.pkl', 'rb') as f:
    x, y = pickle.load(f)
with open('raw/2_feature_rf_testing_normalized.pkl', 'rb') as f:
    x_test, y_test = pickle.load(f)

# %% Fit on training set.
k_fold = KFold(shuffle=True, random_state=924)  # RS for cross validation
optimizer = BayesSearchCV(
    XGBRegressor(n_estimators=100, verbosity=0, n_jobs=-1, gpu_id=0, random_state=502383),
    {
        'learning_rate': Real(0.001, 1, prior='log-uniform'),
        'reg_alpha': Real(0, 1, prior='uniform'),  # L1 regularization
        'reg_lambda': Real(0, 1, prior='uniform'),  # L2 regularization
        'max_depth': (2, 20),
    },
    n_iter=32, cv=k_fold.split(x), random_state=743,
)
optimizer.fit(x, y.ravel())
score = optimizer.score(x, y.ravel())
print(score)

# %% Predict on testing set.
y_test_hat = optimizer.predict(x_test)
score_test = optimizer.score(x_test, y_test)
print(score_test)

# %% Export.
with open('raw/6_feature_rf_estimator_xgb.pkl', 'wb') as f:
    pickle.dump({
        'Estimator': optimizer.best_estimator_, 'R^2': score,
    }, f)
with open('raw/6_feature_rf_estimator_xgb_prediction.pkl', 'wb') as f:
    pickle.dump(y_test_hat, f)
