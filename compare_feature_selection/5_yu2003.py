"""
Reference:
[1] Yu, Lei, and Huan Liu. "Feature selection for high-dimensional data: A fast correlation-based filter solution."
    Proceedings of the 20th international conference on machine learning (ICML-03). 2003.
"""
import numpy as np
import pickle


def entropy_d(x0):
    _, uc = np.unique(x0, return_counts=True, axis=0)
    uf = uc / x0.shape[0]
    return - np.sum(uf * np.log2(uf))


# %% Load data
with open('raw/2_cs_training_normalized.pkl', 'rb') as f:
    x, y = pickle.load(f)
feature_names = x.columns
x_mat, y_mat = x.values, y.flatten()
with open('raw/2_cs_testing_normalized.pkl', 'rb') as f:
    x_test, y_test = pickle.load(f)

# %% Fast correlation-based feature selection
n_features = x_mat.shape[1]
t1 = np.zeros((n_features, 2), dtype='object')
y_entropy = entropy_d(y_mat)
x_entropy = np.zeros(shape=n_features)

for i in np.arange(n_features):
    x_entropy[i] = entropy_d(x_mat[:, i])
    t1[i, 0] = i
    xy = np.stack([x_mat[:, i], y_mat], axis=1)
    t1[i, 1] = 2 * (1 - entropy_d(xy) / (x_entropy[i] + y_entropy))  # SU_ic

s_list = t1[np.argsort(-t1[:, 1])]  # Sorting t1 by SU_ic descending order
# The evidence of removing a particular feature must be prior than itself (has larger SU_ic).
# If the particular feature itself is remained after filtering by theta, the evidence of removing it must be
# remained as well because of larger SU_ic.
# So, for multiple theta, we're able to traverse and remove features before filtering by theta when descending sorted.

s_list_mask = np.ones(shape=n_features, dtype='bool')
for j in np.arange(n_features):
    if s_list_mask[j] == 0:
        continue
    f_j = x_mat[:, s_list[j, 0]]
    for i in np.arange(j + 1, n_features):
        if s_list_mask[i] == 0:
            continue
        f_i = x_mat[:, s_list[i, 0]]
        xy = np.stack([f_i, f_j], axis=1)
        su_ij = 2 * (1 - entropy_d(xy) / (x_entropy[s_list[i, 0]] + x_entropy[s_list[j, 0]]))
        if su_ij >= s_list[i, 1]:
            s_list_mask[i] = 0
s_list = s_list[s_list_mask, :]

# %% Export results.
selected_features = feature_names[list(s_list[:, 0])]
print(selected_features)
with open('raw/5_feature_importance_yu2003.pkl', 'wb') as f:
    pickle.dump(dict(zip(selected_features, s_list[:, 1])), f)
with open('raw/5_feature_yu2003_training_normalized.pkl', 'wb') as f:
    pickle.dump([x[selected_features], y], f)
with open('raw/5_feature_yu2003_testing_normalized.pkl', 'wb') as f:
    pickle.dump([x_test[selected_features], y_test], f)
