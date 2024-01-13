import pickle
from sklearn.ensemble import RandomForestRegressor
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
    RandomForestRegressor(n_jobs=-1, random_state=324, oob_score=True, min_samples_leaf=0.05),
    # Hyperparameter space based on previous decision tree results
    {
        'max_depth': (2, 20),
        'ccp_alpha': Real(0, 0.028, prior='uniform'),
    },
    n_iter=32, n_jobs=-1, cv=k_fold.split(x), random_state=743,  # RS for searching
)
optimizer.fit(x, y.ravel())
score = optimizer.score(x, y.ravel())
print(score)

# %% Predict on testing set.
y_test_hat = optimizer.predict(x_test)
score_test = optimizer.score(x_test, y_test)
print(score_test)

# %% Export.
with open('raw/6_feature_rf_estimator_rf.pkl', 'wb') as f:
    pickle.dump({
        'Estimator': optimizer.best_estimator_, 'R^2': score,
    }, f)
with open('raw/6_feature_rf_estimator_rf_prediction.pkl', 'wb') as f:
    pickle.dump(y_test_hat, f)
