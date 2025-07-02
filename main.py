
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

from data_loader import MarineDataLoader
from models import LSTMModel, GRUModel, TCNModel, TransformerModel
from anomaly_detection import AnomalyDetector, ForecastingAnomalyDetector, Autoencoder, VAE
from uncertainty import MCDropoutModel, ProbabilisticForecaster
from visualization import plot_time_series_with_anomalies, plot_prediction_confidence_intervals
from visualization import visualize_results

from anomaly_detection import perform_anomaly_detection

# Ensure reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- Configuration --- #
data_path = r"D:\Deep Learning System for Extreme Marine Event Detection\extreme_marine_events\data\MetO-NWS-WAV-RAN_1751155418549.nc" # Replace with your actual data path
seq_length = 20

# Model parameters - will be updated after data loading
input_size = None
output_size = None
dropout_rate = 0.2

# Training parameters
num_epochs = 60
batch_size = 32
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Loading and Preprocessing --- #
print("\n--- Loading and Preprocessing Data ---")

data_loader_obj = MarineDataLoader(
    data_path=data_path,
    features=None, # Let MarineDataLoader dynamically detect features
    seq_length=seq_length,
    train_split=0.8
)

# Update input_size and output_size based on dynamically detected features
features = data_loader_obj.features
input_size = len(features)
output_size = len(features)

X_train, y_train, X_test, y_test = data_loader_obj.get_train_test_data()

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"Dynamically detected features: {features}")

# --- Model Training (Forecasting) --- #
# Uncomment ONE of the following model definitions to choose the architecture
# SELECT MODEL: Options "LSTM", "GRU", "TCN", "Transformer"
selected_model = "TCN"  

if selected_model == "LSTM":
    print("\n--- Training Forecasting Model (LSTM) ---")
    
    # LSTM/GRU specific parameters
    hidden_size_lstm_gru = 64
    num_layers_lstm_gru = 2

    model = LSTMModel(input_size, hidden_size_lstm_gru, num_layers_lstm_gru, output_size, dropout=dropout_rate).to(device)

elif selected_model == "GRU":
    print("\n--- Training Forecasting Model (GRU) ---")

    # LSTM/GRU specific parameters
    hidden_size_lstm_gru = 64
    num_layers_lstm_gru = 2

    model = GRUModel(input_size, hidden_size_lstm_gru, num_layers_lstm_gru, output_size, dropout=dropout_rate).to(device)

elif selected_model == "TCN":
    print("\n--- Training Forecasting Model (TCN) ---")
    
    # TCN specific parameters
    num_channels_tcn = [64, 64, 64] # Example: 3 layers with 64 channels each
    kernel_size_tcn = 2

    model = TCNModel(input_size, output_size, num_channels_tcn, kernel_size=kernel_size_tcn, dropout=dropout_rate).to(device)

elif selected_model == "Transformer":
    print("\n--- Training Forecasting Model (Transformer) ---")

    # Transformer specific parameters
    num_heads_transformer = 2 # Number of attention heads
    dim_feedforward_transformer = 128 # Dimension of the feedforward network model
    num_encoder_layers_transformer = 2 # Number of encoder layers

    model = TransformerModel(input_size, num_heads_transformer, num_encoder_layers_transformer, dim_feedforward_transformer, output_size, dropout=dropout_rate).to(device)

else:
    raise ValueError("Invalid model selection! Choose from 'LSTM', 'GRU', 'TCN', 'Transformer'.")

# Optimizer functions same for all models
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}' )

#---------------------------
# --- Anomaly Detection --- #
# Select of the following anomaly detection methods
#---------------------------
# Options # "forecasting" or 'autoencoder' or 'vae'

# Anomaly Detection parameters
# Autoencoder/VAE specific parameters
latent_dim_ae_vae = 16

anomaly_detection_method = "forecasting"  

anomalies_detected = perform_anomaly_detection(
    method=anomaly_detection_method,
    model=model,
    X_train=X_train,
    X_test=X_test,
    y_test=y_test,
    device=device,
    batch_size=batch_size,
    seq_length=seq_length,
    input_size=input_size,
    latent_dim=latent_dim_ae_vae,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    dropout_rate=dropout_rate
)

# --- Uncertainty Quantification (Monte Carlo Dropout) --- #
print("\n--- Quantifying Uncertainty with Monte Carlo Dropout ---")
mc_dropout_model = MCDropoutModel(model, dropout_rate=dropout_rate)
prob_forecaster = ProbabilisticForecaster(mc_dropout_model, device)

# Get mean predictions and standard deviations for the test set
mean_preds, std_preds = prob_forecaster.predict_with_uncertainty(X_test, n_samples=100)

print(f"Mean Predictions shape: {mean_preds.shape}")
print(f"Standard Deviations shape: {std_preds.shape}")

# --- Visualization --- #
visualize_results(data_loader_obj, y_test, mean_preds, std_preds, anomalies_detected, features)
