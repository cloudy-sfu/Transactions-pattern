import sqlite3
import pandas as pd
import numpy as np
import pickle
import multiprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import get_range_values_per_column, impute_dataframe_range

# %% Load data.
connection = sqlite3.connect('data/transactions.db')
with open('sql/melted_time_series.sql', 'r') as f:
    transactions = pd.read_sql(f.read(), connection)
with open('sql/scores.sql', 'r') as f:
    scores = pd.read_sql(f.read(), connection)

# %% time series dataset
x_ts = np.stack([
    transactions.pivot(index='customer_id', columns='period', values=varname).values
    for varname in ['ln_weight', 'price', 'quality']
], axis=2)
_ = np.nan_to_num(x_ts, copy=False, nan=0)
y = scores[['score']].values
x_ts_train, x_ts_test, y_train, y_test = train_test_split(x_ts, y, train_size=0.8, shuffle=True, random_state=324)
# Train
x_ts_scaler = RobustScaler(quantile_range=(0, 95))
x_ts_train_std = x_ts_scaler.fit_transform(x_ts_train.reshape(-1, x_ts_train.shape[-1])).reshape(x_ts_train.shape)
y_scaler = StandardScaler()
y_train_std = y_scaler.fit_transform(y_train)
# Test
x_ts_test_std = x_ts_scaler.transform(x_ts_test.reshape(-1, x_ts_test.shape[-1])).reshape(x_ts_test.shape)
y_test_std = y_scaler.transform(y_test)

# %% cross-sectional dataset
# melt
index = scores[['customer_id']].values
x_cs = np.reshape(x_ts, (x_ts.shape[0] * x_ts.shape[1], x_ts.shape[2]))
# extract features
# Reference:
# [1] Christ, Maximilian, Andreas W. Kempa-Liehr, and Michael Feindt. "Distributed and parallel time series feature
#     extraction for industrial big data applications." arXiv preprint arXiv:1610.07717 (2016).
x_cs = np.hstack([
    np.repeat(index, x_ts.shape[1], axis=0),
    np.tile(np.arange(x_ts.shape[1])[:, np.newaxis], (x_ts.shape[0], 1)),
    x_cs
])
x_cs = pd.DataFrame(data=x_cs, columns=['customer_id', 'period', 'ln_weight', 'price', 'quality'])
x_cs = x_cs.astype(dtype={'customer_id': str, 'period': int, 'ln_weight': float, 'price': float,
                          'quality': float})
x_cs = extract_features(x_cs, column_id='customer_id', column_sort='period', n_jobs=multiprocessing.cpu_count())
x_cs_train, x_cs_test = train_test_split(x_cs, train_size=0.8, shuffle=True, random_state=324)
# Train
col_to_max, col_to_min, col_to_median = get_range_values_per_column(x_cs_train)  # impute
x_cs_train = impute_dataframe_range(x_cs_train, col_to_max, col_to_min, col_to_median)
x_cs_train = x_cs_train.loc[:, x_cs_train.std() > 1e-10]  # remove constants
filtered_features = x_cs_train.columns
x_cs_scaler = RobustScaler(quantile_range=(5, 95))  # normalization
x_cs_train_std = x_cs_scaler.fit_transform(x_cs_train)
x_cs_train_std = pd.DataFrame(data=x_cs_train_std, columns=filtered_features)
# Test
x_cs_test = x_cs_test[filtered_features]
x_cs_test_std = x_cs_scaler.transform(x_cs_test)
x_cs_test_std = pd.DataFrame(data=x_cs_test_std, columns=filtered_features)

# %% Export.
with open('raw/2_ts_scaler.pkl', 'wb') as f:
    pickle.dump(x_ts_scaler, f)
with open('raw/2_cs_scaler.pkl', 'wb') as f:
    pickle.dump(x_cs_scaler, f)
with open('raw/2_y_scaler.pkl', 'wb') as f:
    pickle.dump(y_scaler, f)
with open('raw/2_ts_training_normalized.pkl', 'wb') as f:
    pickle.dump([x_ts_train_std, y_train_std], f)
with open('raw/2_ts_testing_normalized.pkl', 'wb') as f:
    pickle.dump([x_ts_test_std, y_test_std], f)
with open('raw/2_cs_training_normalized.pkl', 'wb') as f:
    pickle.dump([x_cs_train_std, y_train_std], f)
with open('raw/2_cs_testing_normalized.pkl', 'wb') as f:
    pickle.dump([x_cs_test_std, y_test_std], f)
