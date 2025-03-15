from keras.models import load_model, Sequential
from keras.layers import GRU, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import os

figures_folder = 'figures'
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder)

def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def build_model(n_units=64, input_shape=(7, 1), dropout_rate=0.12):
    model = Sequential([
        Bidirectional(GRU(n_units, activation='tanh'),
                      input_shape=input_shape),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

dataframe = pd.read_csv('cleaned_WL.csv', usecols=[1])
dataset = dataframe.values.astype('float32')
train_raw, val_raw = train_test_split(dataset, test_size=0.2, shuffle=False)
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_raw)
val_scaled = scaler.transform(val_raw)
n_steps = 7
X_train, y_train = split_sequence(train_scaled, n_steps)
X_val, y_val = split_sequence(val_scaled, n_steps)
X_train = X_train.reshape((X_train.shape[0], n_steps, 1))
X_val = X_val.reshape((X_val.shape[0], n_steps, 1))
model = build_model(input_shape=(n_steps, 1))
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint('biGRU_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
                    callbacks=[early_stopping, model_checkpoint], verbose=1)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(figures_folder, 'training_validation_loss_biGRU.png'))
plt.close()
model = load_model('biGRU_model.keras')
train_predictions = model.predict(X_train)
val_predictions = model.predict(X_val)
train_predictions = scaler.inverse_transform(train_predictions)
val_predictions = scaler.inverse_transform(val_predictions)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_val_inv = scaler.inverse_transform(y_val.reshape(-1, 1))

def calculate_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_nse(y_true, y_pred):
    return 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

def calculate_re(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / y_true)

train_mse = calculate_mse(y_train_inv, train_predictions)
train_rmse = calculate_rmse(y_train_inv, train_predictions)
train_nse = calculate_nse(y_train_inv, train_predictions)
train_r2 = r2_score(y_train_inv, train_predictions)
train_re = calculate_re(y_train_inv, train_predictions)
val_mse = calculate_mse(y_val_inv, val_predictions)
val_rmse = calculate_rmse(y_val_inv, val_predictions)
val_nse = calculate_nse(y_val_inv, val_predictions)
val_r2 = r2_score(y_val_inv, val_predictions)
val_re = calculate_re(y_val_inv, val_predictions)
print("Training Data Metrics:")
print(f"'MSE': {train_mse}, 'RMSE': {train_rmse}, 'NSE': {train_nse}, 'R2': {train_r2}, 'RE': {train_re}")
print("\nValidation Data Metrics:")
print(f"'MSE': {val_mse}, 'RMSE': {val_rmse}, 'NSE': {val_nse}, 'R2': {val_r2}, 'RE': {val_re}")
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(y_train_inv.flatten(), label='Actual')
plt.plot(train_predictions.flatten(), label='Predicted')
plt.title('Training Data: Actual vs Predicted')
plt.xlabel('Sample')
plt.ylabel('Water Level')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(y_val_inv.flatten(), label='Actual')
plt.plot(val_predictions.flatten(), label='Predicted')
plt.title('Validation Data: Actual vs Predicted')
plt.xlabel('Sample')
plt.ylabel('Water Level')
plt.legend()
plt.savefig(os.path.join(figures_folder, 'actual_vs_predicted_biGRU.png'))
plt.close()
metrics_file = os.path.join(figures_folder, 'model_metrics_biGRU.txt')
with open(metrics_file, 'w') as f:
    f.write("Training Data Metrics:\n")
    f.write(f"'MSE': {train_mse}, 'RMSE': {train_rmse}, 'NSE': {train_nse}, 'R2': {train_r2}, 'RE': {train_re}\n")
    f.write("\nValidation Data Metrics:\n")
    f.write(f"'MSE': {val_mse}, 'RMSE': {val_rmse}, 'NSE': {val_nse}, 'R2': {val_r2}, 'RE': {val_re}\n")
