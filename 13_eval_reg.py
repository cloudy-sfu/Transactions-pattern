import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

# %% Load data.
with open('raw/1_ts_testing_normalized.pkl', 'rb') as f:
    _, y_test_std = pickle.load(f)
with open('raw/1_y_scaler.pkl', 'rb') as f:
    y_scaler = pickle.load(f)
y_test = y_scaler.inverse_transform(y_test_std)

# %% Function
def main(predictions, output_path):
    scores = []
    for method, filepath in predictions.items():
        with open(filepath, 'rb') as f:
            y_test_hat_std = pickle.load(f)
        if len(y_test_hat_std.shape) == 1:
            y_test_hat_std = y_test_hat_std[:, np.newaxis]
        y_test_hat = y_scaler.inverse_transform(y_test_hat_std)
        # scores
        r2 = r2_score(y_test, y_test_hat)  # R^2
        mse = mean_squared_error(y_test, y_test_hat)  # MSE
        scores.append({
            'Method': method,
            'R^2': r2,
            'MSE': mse,
        })
    scores = pd.DataFrame(scores)
    scores.to_csv(output_path, index=False)

# %% Compare hybrid, time series, and cross-sectional methods.
estimations = {
    'Hybrid': 'raw/10_feature_rf_hybrid_prediction.pkl',
    'Time series': 'raw/4_estimator_lstm_prediction.pkl',
    'Cross-sectional': 'raw/9_feature_rf_stacking_prediction.pkl',
}
scores_path = 'results/13_testing_r2_hybrid.csv'
main(estimations, scores_path)

# %% Compare base models and stacking.
estimations = {
    'GBDT': 'raw/6_feature_rf_estimator_gbdt_prediction.pkl',
    'Random forest': 'raw/6_feature_rf_estimator_rf_prediction.pkl',
    'SVR': 'raw/6_feature_rf_estimator_svr_prediction.pkl',
    'XGBoost': 'raw/6_feature_rf_estimator_xgb_prediction.pkl',
    'Stacking': 'raw/9_feature_rf_stacking_prediction.pkl',
}
scores_path = 'results/13_testing_r2_stacking.csv'
main(estimations, scores_path)

# %% Compare feature selection methods.
estimations = {
    'yu2003': 'raw/12_feature_yu2003_stacking_prediction.pkl',
    'Random forest': 'raw/9_feature_rf_stacking_prediction.pkl',
}
scores_path = 'results/13_testing_r2_features.csv'
main(estimations, scores_path)
