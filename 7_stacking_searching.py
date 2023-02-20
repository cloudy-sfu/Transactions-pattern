from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
import pickle
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split

# %% Load data.
with open('raw/2_feature_rf_training_normalized.pkl', 'rb') as f:
    x, y = pickle.load(f)
x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=974238)
with open('raw/2_feature_rf_testing_normalized.pkl', 'rb') as f:
    x_test, y_test = pickle.load(f)

# %% Load models.
base_models = []
base_model_names = ['deep_forest', 'gbdt', 'rf', 'svr', 'xgb']
for name in base_model_names:
    with open(f'raw/6_feature_rf_estimator_{name}.pkl', 'rb') as f:
        estimator = pickle.load(f)['Estimator']
    base_models.append((name, estimator))

# %% Build stacking models.
def traverse(n):
    def _t(s):
        s_str = bin(s)[2:]
        pad = n - len(s_str)
        return [False] * pad + [i == '1' for i in s_str]
    return list(map(_t, range(1, 2 ** n)))

final_models = {
    'lr': LinearRegression(),
    'elastic_net': ElasticNetCV(),
    'lasso': LassoCV()
}
n_ = len(base_models)
used_models_id = traverse(n_)
k_fold = KFold(shuffle=True, random_state=924)  # RS for cross validation
results = []
models = []
for name, final_model in final_models.items():
    print(name)
    for comb in tqdm(used_models_id):
        used_models = [base_models[i] for i in range(n_) if comb[i]]
        stacking = StackingRegressor(used_models, final_estimator=final_model, n_jobs=-1,
                                     cv=k_fold.split(x_train))
        stacking.fit(x_train, y_train.ravel())
        results.append(
            dict(zip(base_model_names, comb)) | {
                'method': name, 'training_r2': stacking.score(x_train, y_train.ravel()),
                'validation_r2': stacking.score(x_valid, y_valid.ravel())}
        )
        # Refit with all training & validation set, for better coverage of data. The cost is doubled training time.
        # This approach should be only used for model selection (research).
        stacking = StackingRegressor(used_models, final_estimator=final_model, n_jobs=-1,
                                     cv=k_fold.split(x))
        stacking.fit(x, y.ravel())
        # Make stacking model pickleable. The cost is that, if reload the model from the pickled file, we should
        # program the same k_fold generator again.
        stacking.cv = k_fold.n_splits
        models.append(stacking)

# %% Export.
with open('raw/7_feature_rf_stacking.pkl', 'wb') as f:
    pickle.dump(models, f)
with open('raw/7_feature_rf_stacking_r2.pkl', 'wb') as f:
    pickle.dump(results, f)
