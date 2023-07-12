import pickle
from sklearn.metrics import r2_score
import tensorflow as tf
from sklearn.model_selection import train_test_split

# %% Load data.
with open('raw/1_ts_training_normalized.pkl', 'rb') as f:
    x, y = pickle.load(f)
with open('raw/1_ts_testing_normalized.pkl', 'rb') as f:
    x_test, y_test = pickle.load(f)

# %% Train and test.
l0 = tf.keras.layers.Input(shape=(x.shape[1], x.shape[2]))
_, l1_h_t, _ = tf.keras.layers.LSTM(64, return_state=True)(l0)
l2 = tf.keras.layers.Dense(128, activation='relu')(l1_h_t)
l3 = tf.keras.layers.Dense(128, activation='relu')(l2)
l5 = tf.keras.layers.Dense(32, activation='relu')(l3)
l6 = tf.keras.layers.Dense(1, activation='linear')(l5)
my_model = tf.keras.Model(l0, l6)
my_model.compile(optimizer='adam', loss='mse')
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f'raw/4_tensorboard/')
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=60)
save_best = tf.keras.callbacks.ModelCheckpoint('raw/4_lstm.h5', monitor='val_loss', save_best_only=True)
x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=974238)
history = my_model.fit(
    x_train, y_train, validation_data=(x_valid, y_valid),
    epochs=1000, batch_size=1200, callbacks=[stop_early, save_best, tensorboard]
)
y_hat = my_model.predict(x)
score_train = r2_score(y, y_hat)
print(score_train)

# %% Predict on testing set.
y_test_hat = my_model.predict(x_test)
score_test = r2_score(y_test, y_test_hat)
print(score_test)

# %% Export.
with open('raw/4_estimator_lstm.pkl', 'wb') as f:
    pickle.dump({
        'training_history': history.history,
        'R^2': score_train,
    }, f)
with open('raw/4_estimator_lstm_prediction.pkl', 'wb') as f:
    pickle.dump(y_test_hat, f)
