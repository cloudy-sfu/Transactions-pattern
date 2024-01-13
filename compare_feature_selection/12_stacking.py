import pickle
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from skopt import BayesSearchCV
from skopt.space import Real
from sklearn.model_selection import KFold

# %% Load data.
with open('raw/5_feature_yu2003_training_normalized.pkl', 'rb') as f:
    x, y = pickle.load(f)
with open('raw/5_feature_yu2003_testing_normalized.pkl', 'rb') as f:
    x_test, y_test = pickle.load(f)

# %% Build model on training set.
k_fold = KFold(shuffle=True, random_state=924)  # RS for cross validation
stacking = StackingRegressor([
    # ('gbdt', GradientBoostingRegressor(random_state=502383, min_samples_leaf=0.05)),
    ('svr', SVR(kernel='rbf', max_iter=5000, cache_size=2048, epsilon=0)),
    ('xgboost', XGBRegressor(n_estimators=100, verbosity=0, n_jobs=-1, gpu_id=0, random_state=502383)),
], final_estimator=LinearRegression(), n_jobs=-1)
optimizer = BayesSearchCV(stacking, {
    # 'gbdt__learning_rate': Real(0.001, 1, prior='log-uniform'),
    # 'gbdt__ccp_alpha': Real(0, 0.028, prior='uniform'),
    'svr__C': Real(0.001, 10, prior='log-uniform'),
    'xgboost__learning_rate': Real(0.001, 1, prior='log-uniform'),
    'xgboost__reg_alpha': Real(0, 1, prior='uniform'),  # L1 regularization
    'xgboost__reg_lambda': Real(0, 1, prior='uniform'),  # L2 regularization
    'xgboost__max_depth': (2, 20),
}, n_iter=96, cv=k_fold.split(x), random_state=743)
optimizer.fit(x, y.ravel())
score = optimizer.score(x, y.ravel())
print(score)

# %% Predict on testing set.
y_test_hat = optimizer.predict(x_test)
score_test = optimizer.score(x_test, y_test.ravel())
print(score_test)

# %% Export.
with open('raw/12_feature_yu2003_stacking_prediction.pkl', 'wb') as f:
    pickle.dump(y_test_hat, f)
