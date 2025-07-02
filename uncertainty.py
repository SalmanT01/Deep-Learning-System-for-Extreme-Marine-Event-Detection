
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm

class MCDropoutModel(nn.Module):
    def __init__(self, base_model, dropout_rate=0.5):
        super(MCDropoutModel, self).__init__()
        self.base_model = base_model
        self.dropout_rate = dropout_rate

        # Apply dropout to all dropout layers in the base model
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.dropout_rate
                module.train() # Ensure dropout is active during inference

    def forward(self, x, n_samples=100):
        predictions = []
        for _ in range(n_samples):
            predictions.append(self.base_model(x))
        return torch.stack(predictions)

class ProbabilisticForecaster:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict_with_uncertainty(self, data, n_samples=100):
        self.model.eval() # Ensure dropout layers are active
        with torch.no_grad():
            data = data.to(self.device)
            mc_predictions = self.model(data, n_samples=n_samples)
        
        mean_predictions = mc_predictions.mean(dim=0)
        std_predictions = mc_predictions.std(dim=0)
        
        return mean_predictions.cpu().numpy(), std_predictions.cpu().numpy()

    def get_alert_thresholds(self, mean_preds, std_preds, confidence_level=0.95):
        # Calculate Z-score for the desired confidence level
        # For a two-tailed test, alpha = 1 - confidence_level, and we look for alpha/2
        z_score = norm.ppf(1 - (1 - confidence_level) / 2)
        
        upper_bound = mean_preds + z_score * std_preds
        lower_bound = mean_preds - z_score * std_preds
        return upper_bound, lower_bound

    def generate_alerts(self, actual_values, mean_preds, std_preds, confidence_level=0.95):
        upper_bound, lower_bound = self.get_alert_thresholds(mean_preds, std_preds, confidence_level)
        
        # Anomaly if actual value falls outside the prediction interval
        alerts = (actual_values > upper_bound) | (actual_values < lower_bound)
        return alerts

if __name__ == '__main__':
    # Dummy setup for testing
    input_size = 3
    sequence_length = 20
    hidden_size = 64
    num_layers = 2
    output_size = 3
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assuming you have a base model (e.g., LSTM) with dropout layers
    from models import LSTMModel
    base_lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout=0.2).to(device)

    # Wrap it with MCDropoutModel
    mc_dropout_model = MCDropoutModel(base_lstm_model, dropout_rate=0.2)

    # Create a dummy input and dummy actual values for testing
    dummy_input = torch.randn(batch_size, sequence_length, input_size)
    dummy_actual_values = torch.randn(batch_size, output_size).cpu().numpy() # Simulate actual observed values

    # Initialize ProbabilisticForecaster
    forecaster = ProbabilisticForecaster(mc_dropout_model, device)

    # Predict with uncertainty
    mean_preds, std_preds = forecaster.predict_with_uncertainty(dummy_input, n_samples=100)

    print(f"Mean Predictions shape: {mean_preds.shape}")
    print(f"Standard Deviations shape: {std_preds.shape}")

    # Get alert thresholds
    upper, lower = forecaster.get_alert_thresholds(mean_preds, std_preds, confidence_level=0.95)
    print(f"Upper bound shape: {upper.shape}")
    print(f"Lower bound shape: {lower.shape}")

    # Generate alerts
    alerts = forecaster.generate_alerts(dummy_actual_values, mean_preds, std_preds, confidence_level=0.95)
    print(f"Alerts shape: {alerts.shape}")
    print(f"Number of detected alerts: {np.sum(alerts)}")


