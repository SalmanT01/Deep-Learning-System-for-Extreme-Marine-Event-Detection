
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
#from models import Autoencoder, VAE  # adjust import paths as needed
#from main import ForecastingAnomalyDetector, AnomalyDetector  # adjust paths as needed


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the input for the linear layers
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)  # Mean
        self.fc22 = nn.Linear(latent_dim, latent_dim)  # Log variance
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc4 = nn.Linear(latent_dim, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the input
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class AnomalyDetector:
    def __init__(self, model, criterion, threshold=None):
        self.model = model
        self.criterion = criterion
        self.threshold = threshold

    def train(self, data_loader, num_epochs, learning_rate, device):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()
        for epoch in range(num_epochs):
            for batch_data in data_loader:
                # Handle cases where DataLoader yields a single tensor or a tuple
                if isinstance(batch_data, (list, tuple)):
                    data = batch_data[0].to(device)
                    target = batch_data[1].to(device) if len(batch_data) > 1 else data # If only one item, assume input is target
                else:
                    data = batch_data.to(device)
                    target = data # For AE/VAE, input is also the target

                optimizer.zero_grad()
                
                if isinstance(self.model, VAE):
                    recon_batch, mu, logvar = self.model(data)
                    # VAE criterion should return a scalar loss
                    loss = self.criterion(recon_batch, data, mu, logvar)
                else:
                    output = self.model(data)
                    # Ensure loss is a scalar for backward pass
                    loss = self.criterion(output, target.view(target.size(0), -1)).mean()
                
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

    def predict_error(self, data, device):
        self.model.eval()
        with torch.no_grad():
            data = data.to(device)
            if isinstance(self.model, VAE):
                recon_data, _, _ = self.model(data)
                reconstruction_error = torch.mean((recon_data - data.view(data.size(0), -1))**2, dim=1)
            else:
                output = self.model(data)
                # The criterion is nn.MSELoss(reduction='none'), so output is element-wise error
                reconstruction_error = self.criterion(output, data.view(data.size(0), -1)).mean(dim=1)
        return reconstruction_error.cpu().numpy()

    def set_threshold(self, errors, percentile=95):
        self.threshold = np.percentile(errors, percentile)
        print(f"Anomaly threshold set to: {self.threshold:.4f}")

    def detect(self, errors):
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")
        return errors > self.threshold

class ForecastingAnomalyDetector:
    def __init__(self, forecasting_model, criterion, threshold=None):
        self.forecasting_model = forecasting_model
        self.criterion = criterion
        self.threshold = threshold

    def predict_error(self, data_loader, device):
        self.forecasting_model.eval()
        prediction_errors = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = self.forecasting_model(inputs)
                # Assuming targets are the next sequence element, and predictions match its shape
                error = self.criterion(predictions, targets).cpu().numpy()
                prediction_errors.append(error)
        return np.concatenate(prediction_errors)

    def set_threshold(self, errors, percentile=95):
        self.threshold = np.percentile(errors, percentile)
        print(f"Forecasting Anomaly threshold set to: {self.threshold:.4f}")

    def detect(self, errors):
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")
        return errors > self.threshold



# ---------------------------
# --- Anomaly Detection --- #
# ---------------------------

def perform_anomaly_detection(method, model, X_train, X_test, y_test, device, batch_size, seq_length, input_size, latent_dim, num_epochs, learning_rate, dropout_rate):
    """
    Handles anomaly detection using the specified method: 'forecasting', 'autoencoder', 'vae'.
    """
    print(f"\n--- Performing {method.capitalize()}-based Anomaly Detection ---")

    if method == "forecasting":
        anomaly_detector = ForecastingAnomalyDetector(model, nn.MSELoss(reduction='none'))

        anomaly_test_dataset = TensorDataset(X_test, y_test)
        anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=batch_size, shuffle=False)

        errors = anomaly_detector.predict_error(anomaly_test_loader, device)

    elif method == "autoencoder":
        ae_model = Autoencoder(input_size * seq_length, latent_dim).to(device)
        anomaly_detector = AnomalyDetector(ae_model, nn.MSELoss(reduction='none'))

        ae_train_dataset = TensorDataset(X_train.view(X_train.size(0), -1))
        ae_train_loader = DataLoader(ae_train_dataset, batch_size=batch_size, shuffle=True)

        anomaly_detector.train(ae_train_loader, num_epochs, learning_rate, device)

        ae_test_data = X_test.view(X_test.size(0), -1)
        errors = anomaly_detector.predict_error(ae_test_data, device)

    elif method == "vae":
        vae_model = VAE(input_size * seq_length, latent_dim).to(device)

        def vae_loss_function(recon_x, x, mu, logvar):
            BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, input_size * seq_length), reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - mu.exp())
            return BCE + KLD

        anomaly_detector = AnomalyDetector(vae_model, vae_loss_function)

        vae_train_dataset = TensorDataset(X_train.view(X_train.size(0), -1))
        vae_train_loader = DataLoader(vae_train_dataset, batch_size=batch_size, shuffle=True)

        anomaly_detector.train(vae_train_loader, num_epochs, learning_rate, device)

        vae_test_data = X_test.view(X_test.size(0), -1)
        errors = anomaly_detector.predict_error(vae_test_data, device)

    else:
        raise ValueError("Invalid anomaly detection method. Choose from 'forecasting', 'autoencoder', 'vae'.")

    # Common post-processing for all methods
    anomaly_detector.set_threshold(errors, percentile=95)
    anomalies_detected_per_feature = anomaly_detector.detect(errors)

    if method == "forecasting":
        anomalies_detected = np.any(anomalies_detected_per_feature, axis=1)
    else:
        anomalies_detected = anomalies_detected_per_feature

    print(f"Number of anomalies detected: {np.sum(anomalies_detected)}")

    return anomalies_detected



if __name__ == '__main__':
    # Dummy data for testing all anomaly detection methods
    input_dim_flat = 60 # e.g., 20 timesteps * 3 features
    latent_dim_ae_vae = 16
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate some dummy data (e.g., 1000 sequences of 20 timesteps, 3 features each)
    dummy_data = torch.randn(1000, 20, 3)
    # For autoencoder/VAE, we need to flatten the input to (batch_size, sequence_length * num_features)
    dummy_data_flat = dummy_data.view(dummy_data.size(0), -1)

    # Create a dummy DataLoader for AE/VAE training (input and target are the same)
    dummy_ae_vae_dataset = TensorDataset(dummy_data_flat, dummy_data_flat) 
    dummy_ae_vae_loader = DataLoader(dummy_ae_vae_dataset, batch_size=batch_size, shuffle=True)

    # Create dummy data for forecasting (input sequence, target next step)
    # Assuming input_dim for forecasting model is sequence_length * num_features
    # And output_dim is num_features (predicting the next single time step)
    forecasting_input_size = dummy_data.shape[1] * dummy_data.shape[2] # 20 * 3 = 60
    forecasting_output_size = dummy_data.shape[2] # 3

    dummy_forecasting_inputs = dummy_data[:, :-1, :].reshape(dummy_data.shape[0], -1) # All but last timestep, flattened
    dummy_forecasting_targets = dummy_data[:, -1, :] # Last timestep

    dummy_forecasting_dataset = TensorDataset(dummy_forecasting_inputs, dummy_forecasting_targets)
    dummy_forecasting_loader = DataLoader(dummy_forecasting_dataset, batch_size=batch_size, shuffle=False)

    # --- Anomaly Detection Methods --- #
    # Uncomment ONE of the following blocks to test a specific anomaly detection method

    # 1. Forecasting-based Anomaly Detection
    # print("\n--- Testing Forecasting-based Anomaly Detection ---")
    # # Dummy forecasting model for testing
    # class DummyForecastingModel(nn.Module):
    #     def __init__(self, input_size, output_size):
    #         super().__init__()
    #         self.linear = nn.Linear(input_size, output_size)
    #     def forward(self, x):
    #         # Flatten the input for this dummy linear model
    #         return self.linear(x.view(x.size(0), -1))

    # forecasting_model = DummyForecastingModel(forecasting_input_size, forecasting_output_size).to(device)
    # forecasting_criterion = nn.MSELoss(reduction='none')
    # forecasting_anomaly_detector = ForecastingAnomalyDetector(forecasting_model, forecasting_criterion)

    # # Train the dummy forecasting model (very basic training for demonstration)
    # optimizer_forecast = torch.optim.Adam(forecasting_model.parameters(), lr=learning_rate)
    # forecasting_model.train()
    # for epoch in range(num_epochs):
    #     for batch_idx, (inputs, targets) in enumerate(dummy_forecasting_loader):
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         optimizer_forecast.zero_grad()
    #         predictions = forecasting_model(inputs)
    #         loss = torch.mean(forecasting_criterion(predictions, targets))
    #         loss.backward()
    #         optimizer_forecast.step()
    #     print(f'Forecast Model Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

    # print("Predicting forecasting errors...")
    # errors = forecasting_anomaly_detector.predict_error(dummy_forecasting_loader, device)
    # print(f"First 5 forecasting errors: {errors[:5]}")

    # forecasting_anomaly_detector.set_threshold(errors, percentile=95)
    # anomalies = forecasting_anomaly_detector.detect(errors)
    # print(f"Number of detected forecasting anomalies: {np.sum(anomalies)}")

    # 2. Autoencoder-based Anomaly Detection (currently active)
    print("\n--- Testing Autoencoder Anomaly Detection ---")
    ae_model = Autoencoder(input_dim_flat, latent_dim_ae_vae).to(device)
    ae_criterion = nn.MSELoss(reduction='none') # Use none to get individual errors
    ae_detector = AnomalyDetector(ae_model, ae_criterion)

    print("Training Autoencoder...")
    ae_detector.train(dummy_ae_vae_loader, num_epochs, learning_rate, device)

    print("Predicting reconstruction errors...")
    errors = ae_detector.predict_error(dummy_data_flat, device)
    print(f"First 5 reconstruction errors: {errors[:5]}")

    ae_detector.set_threshold(errors, percentile=95)
    anomalies = ae_detector.detect(errors)
    print(f"Number of detected anomalies: {np.sum(anomalies)}")

    # 3. VAE-based Anomaly Detection
    # print("\n--- Testing VAE Anomaly Detection ---")
    # vae_model = VAE(input_dim_flat, latent_dim_ae_vae).to(device)
    # # VAE loss combines reconstruction loss and KL divergence
    # def vae_loss_function(recon_x, x, mu, logvar):
    #     BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, input_dim_flat), reduction='sum')
    #     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #     return (BCE + KLD).mean() # Ensure scalar output

    # vae_detector = AnomalyDetector(vae_model, vae_loss_function) # Note: criterion here is a function, not nn.Module

    # print("Training VAE...")
    # # For VAE training, we need to modify the train method slightly to handle mu and logvar
    # # This is a simplified example, a full VAE training loop would be more involved.
    # optimizer_vae = torch.optim.Adam(vae_model.parameters(), lr=learning_rate)
    # vae_model.train()
    # for epoch in range(num_epochs):
    #     for batch_data in dummy_ae_vae_loader:
    #         data = batch_data[0].to(device)
    #         optimizer_vae.zero_grad()
    #         recon_batch, mu, logvar = vae_model(data)
    #         loss = vae_loss_function(recon_batch, data, mu, logvar)
    #         loss.backward()
    #         optimizer_vae.step()
    #     print(f'VAE Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

    # print("Predicting VAE reconstruction errors...")
    # # For VAE, anomaly score can be reconstruction error or negative log-likelihood
    # # Here, we'll use reconstruction error for simplicity, similar to AE.
    # vae_model.eval()
    # with torch.no_grad():
    #     data = dummy_data_flat.to(device)
    #     recon_data, _, _ = vae_model(data)
    #     vae_errors = torch.mean((recon_data - data.view(data.size(0), -1))**2, dim=1).cpu().numpy()

    # print(f"First 5 VAE reconstruction errors: {vae_errors[:5]}")
    # vae_detector.set_threshold(vae_errors, percentile=95)
    # anomalies = vae_detector.detect(vae_errors)
    # print(f"Number of detected VAE anomalies: {np.sum(anomalies)}")


