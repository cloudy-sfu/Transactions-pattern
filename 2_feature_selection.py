import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import pickle
from multiprocessing import Pool
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from skopt.space import Real
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# %% Load data.
with open('raw/1_cs_training_normalized.pkl', 'rb') as f:
    x, y = pickle.load(f)
with open('raw/1_cs_testing_normalized.pkl', 'rb') as f:
    x_test, y_test = pickle.load(f)

# %% Find the range of optimal hyperparameter of random forest.
optimizer = DecisionTreeRegressor(random_state=324)
optimizer.fit(x, y.ravel())
path = optimizer.cost_complexity_pruning_path(x, y.ravel())

def ccp_depth(alpha):
    dt = DecisionTreeRegressor(ccp_alpha=alpha, random_state=324)
    dt.fit(x, y)
    return dt.tree_.max_depth

pool = Pool()
depths = pool.map(ccp_depth, path.ccp_alphas)
path.ccp_alphas = path.ccp_alphas[:-1]
depths = depths[:-1]

# %% Visualize
f, ax = plt.subplots()
ax.plot(path.ccp_alphas, depths, drawstyle="steps-post", marker='o', markersize=3)
for i, ccp_alpha, depth in zip(range(len(depths)), path.ccp_alphas, depths):
    if depth not in depths[:i]:
        ax.annotate(f'({round(ccp_alpha, 4)}, {round(depth, 0)})', xy=(ccp_alpha, depth), textcoords='data')
ax.set_xlabel("CCP alpha")
ax.set_ylabel("Tree's depth")
f.savefig('results/2_ccp_alpha_&_depth.eps')
plt.close(f)
# Result: Search max_depth in random forest in interval [2, 20], and ccp_alpha in interval [0, 0.0279].

# %% Optimize a random forest model to calc feature importance.
k_fold = KFold(shuffle=True, random_state=387)
optimizer = BayesSearchCV(
    RandomForestRegressor(n_jobs=-1, random_state=324, oob_score=True),
    # Hyperparameter space based on previous decision tree results
    {
        'max_depth': (2, 20),
        'ccp_alpha': Real(0, 0.028, prior='uniform'),
    },
    n_iter=32, n_jobs=-1, cv=k_fold.split(x), random_state=743,  # RS for searching
)
optimizer.fit(x, y.ravel())

# %% Feature importance.
feature_importance = pd.DataFrame({'Name': x.columns,
                                   'Importance': optimizer.best_estimator_.feature_importances_})
feature_importance.sort_values(by='Importance', ascending=False, inplace=True)

# %% Select top-k features
max_features = 50
r2 = np.zeros(shape=(5, max_features))
k_fold = KFold(shuffle=True, random_state=387)
x_train, y_train, x_valid, y_valid = [], [], [], []
for (idx_train, idx_valid) in k_fold.split(x):
    x_train.append(x.values[idx_train])
    y_train.append(y[idx_train])
    x_valid.append(x.values[idx_valid])
    y_valid.append(y[idx_valid])
for i in range(max_features):
    k = i + 1
    top_k = feature_importance.index[:k].tolist()
    for j in range(5):
        optimizer_k = RandomForestRegressor(n_jobs=-1, random_state=324, oob_score=True, **optimizer.best_params_)
        optimizer_k.fit(x_train[j][:, top_k], y_train[j].ravel())
        r2[j, i] = optimizer_k.score(x_valid[j][:, top_k], y_valid[j].ravel())
r2 = np.mean(r2, axis=0)
best_k = np.argmax(r2) + 1

# %% Export.
with open('raw/2_feature_selection_model.pkl', 'wb') as f:
    pickle.dump(optimizer.best_estimator_, f)
with open('raw/2_feature_importance.pkl', 'wb') as f:
    pickle.dump(feature_importance, f)
top_best_k = feature_importance.index[:best_k].tolist()
with open('raw/2_feature_rf_training_normalized.pkl', 'wb') as f:
    pickle.dump([x.values[:, top_best_k], y], f)
with open('raw/2_feature_rf_testing_normalized.pkl', 'wb') as f:
    pickle.dump([x_test.values[:, top_best_k], y_test], f)
