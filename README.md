# lahti_deep_learning_water_level_example
Example codes to tune the hyperparameters of deep learning model (bidirectional GRU) as well as model training for time series regression tasks like forecasting lake water levels. 
## Specifications
-**forecast horizon**: 1 day ahead of lake water level of vesijärvi
-**input feature**: 7 days of historical lake water level of vesijärvi (univariant)
-**data source**: Finnish Environment Institute
-**hyperparameters tuned**: batch size, numbers of neurons for each layer, number of layers (consider pure BiGRU layers and mixtures of GRU and BiGRU), activation function, dropout rate, optimizer 
-**epochs** = 100
-**train test split ratio** = 0.2
-**loss function** mean squared loss
-**tuning method**: bayesian optimization using Optuna module
-**scaler**: MinMaxscaler [0-1]
