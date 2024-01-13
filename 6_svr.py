import pickle
from sklearn.svm import SVR
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
    SVR(kernel='rbf', max_iter=5000, cache_size=2048, epsilon=0),
    {'C': Real(0.001, 10, prior='log-uniform')},
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
with open('raw/6_feature_rf_estimator_svr.pkl', 'wb') as f:
    pickle.dump({
        'Estimator': optimizer.best_estimator_, 'R^2': score,
    }, f)
with open('raw/6_feature_rf_estimator_svr_prediction.pkl', 'wb') as f:
    pickle.dump(y_test_hat, f)
