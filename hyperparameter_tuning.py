import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scikeras.wrappers import KerasRegressor
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, Bidirectional
import optuna

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

def build_model(trial, input_shape=(7, 1)):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    model = Sequential()
    bi_gru_forced = False
    for i in range(n_layers):
        if i == n_layers - 1 and not bi_gru_forced:
            layer_type = 'BiGRU'
        else:
            layer_type = trial.suggest_categorical(f'layer_type_{i}', ['GRU', 'BiGRU'])
        if layer_type == 'BiGRU':
            bi_gru_forced = True
        n_units = trial.suggest_categorical(f'n_units_layer_{i}', [16, 32, 64, 128])
        activation = trial.suggest_categorical(f'activation_layer_{i}', ['relu', 'tanh'])
        return_sequences = True if i < n_layers - 1 else False
        if layer_type == 'GRU':
            if i == 0:
                model.add(GRU(units=n_units, activation=activation, input_shape=input_shape,
                              return_sequences=return_sequences))
            else:
                model.add(GRU(units=n_units, activation=activation, return_sequences=return_sequences))
        else:
            if i == 0:
                model.add(Bidirectional(GRU(units=n_units, activation=activation,
                              return_sequences=return_sequences), input_shape=input_shape))
            else:
                model.add(Bidirectional(GRU(units=n_units, activation=activation, return_sequences=return_sequences)))
        dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
    model.compile(optimizer=optimizer, loss='mse')
    return model

def objective(trial):
    model = KerasRegressor(model=lambda: build_model(trial, input_shape=(time_steps, features)), verbose=0)
    model.fit(X_train, y_train, epochs=100, batch_size=trial.suggest_categorical('batch_size', [16, 32, 64]), verbose=0)
    score = model.score(X_val, y_val)
    return score

dataframe = pd.read_csv('cleaned_WL.csv', usecols=[1])
dataset = dataframe.values.astype('float32')
train_raw, val_raw = train_test_split(dataset, test_size=0.2, shuffle=False)
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_raw)
val_scaled = scaler.transform(val_raw)
time_steps, features = 7, 1
X_train, y_train = split_sequence(train_scaled, time_steps)
X_val, y_val = split_sequence(val_scaled, time_steps)
X_train = X_train.reshape((X_train.shape[0], time_steps, features))
X_val = X_val.reshape((X_val.shape[0], time_steps, features))
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=400)
trial = study.best_trial
print("Best trial:")
print("Value:", trial.value)
print("Params:")
for key, value in trial.params.items():
    print("   {}: {}".format(key, value))
best_model = KerasRegressor(model=lambda: build_model(trial, input_shape=(time_steps, features)), verbose=0)
best_model.fit(X_train, y_train, epochs=100, batch_size=trial.params['batch_size'], verbose=0)
test_accuracy = best_model.score(X_val, y_val)
print('Test accuracy:', test_accuracy)
