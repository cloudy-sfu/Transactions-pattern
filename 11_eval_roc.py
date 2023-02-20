import pickle
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

with open('raw/2_ts_testing_normalized.pkl', 'rb') as f:
    _, y_test_std = pickle.load(f)
with open('raw/2_y_scaler.pkl', 'rb') as f:
    y_scaler = pickle.load(f)
y_test = y_scaler.inverse_transform(y_test_std)
thresholds = np.array([43.548, 52.4506, 64.6253])  # columns: is C or higher, is B or A, is A
y_test = y_test >= thresholds

estimations = {
    'yu2003+Stacking': 'raw/12_feature_yu2003_stacking_prediction.pkl',
    'RF+Stacking': 'raw/9_feature_rf_stacking_prediction.pkl',
    'LSTM': 'raw/9_feature_rf_stacking_prediction.pkl',
    'LSTM & RF+Stacking': 'raw/10_feature_rf_hybrid_prediction.pkl',
}
figure_path = 'results/11_testing_roc.eps'

# Function
scores = []
fig, axes = plt.subplots(figsize=(14, 4), ncols=3,
                         gridspec_kw=dict(left=0.05, right=0.98, bottom=0.15, top=0.96, wspace=0.18))
for method, filepath in estimations.items():
    with open(filepath, 'rb') as f:
        y_test_hat_std = pickle.load(f)
    if len(y_test_hat_std.shape) == 1:
        y_test_hat_std = y_test_hat_std[:, np.newaxis]
    y_test_hat = y_scaler.inverse_transform(y_test_hat_std)
    # plot
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_test_hat)
        auc_ = auc(fpr, tpr)
        axes[i].plot(fpr, tpr, label=f'{method} (AUC = {round(auc_, 3)})')
for i in range(3):
    axes[i].legend()
    axes[i].set_xlabel('False Positive Rate')
    axes[i].set_ylabel('True Positive Rate')
fig.savefig(figure_path)
plt.close(fig)
