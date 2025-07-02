# Deep Learning for Extreme Marine Events Detection and Forecasting

## Project Overview
This project implements a deep learning-based system for detecting and forecasting extreme marine events, such as rogue waves and storm surges, using metocean datasets. The system leverages state-of-the-art deep learning methods for anomaly detection in time series data, time series classification to identify patterns preceding extreme events, and probabilistic forecasting to quantify uncertainty in event prediction.

## Features
- **Data Handling**: Loads and preprocesses NetCDF-formatted metocean data (e.g., significant wave height (Hs), peak period (Tp), sea level pressure).
- **Sequence Modeling**: Implements various deep learning architectures for time series modeling, including LSTM, GRU, Temporal Convolutional Networks (TCN), and Transformer-based models.
- **Anomaly Detection**: Incorporates frameworks for anomaly detection, such as Autoencoder-based reconstruction error detection, Variational Autoencoders (VAE), and forecasting-based detection.
- **Uncertainty Quantification**: Quantifies prediction uncertainty using Monte Carlo Dropout.
- **Visualization**: Provides tools for visualizing time series with detected anomalies and model prediction confidence intervals.
- **Reproducibility**: Ensures code reproducibility through seed settings.

## Project Structure
```
extreme_marine_events/
├── data/                 # Stores metocean datasets (e.g., .nc files)
├── models/               # Saved trained models
├── src/                  # Source code for the system
│   ├── data_loader.py    # Handles data loading and preprocessing
│   ├── models.py         # Defines deep learning model architectures (LSTM, GRU, TCN, Transformer)
│   ├── anomaly_detection.py # Implements anomaly detection frameworks (AE, VAE, Forecasting-based)
│   ├── uncertainty.py    # Implements uncertainty quantification (MC Dropout) and probabilistic forecasting
│   ├── visualization.py  # Provides visualization utilities
│   └── main.py           # Main script for integration, training, and demonstration
├── notebooks/            # Jupyter notebooks for experimentation and analysis (optional)
└── README.md             # Project documentation
```

## Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd extreme_marine_events
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install torch numpy matplotlib pandas netCDF4 scikit-learn xarray scipy
    ```

## Usage

To run the main demonstration script:

```bash
python3 src/main.py
```

This script will:
1.  Generate a dummy NetCDF data file if one doesn't exist.
2.  Load and preprocess the data.
3.  Train an LSTM forecasting model.
4.  Perform forecasting-based anomaly detection.
5.  Quantify uncertainty using Monte Carlo Dropout.
6.  Generate and save visualization plots (`time_series_anomalies.png` and `prediction_confidence_intervals.png`) in the current directory.

## Modules Details

### `data_loader.py`
-   `MarineDataLoader`: Class to load NetCDF data, apply min-max scaling, and create time series sequences using a sliding window approach. It also handles splitting data into training and testing sets.

### `models.py`
-   `LSTMModel`, `GRUModel`: Standard Recurrent Neural Network architectures for sequence modeling.
-   `TCNModel`: Implementation of a Temporal Convolutional Network, suitable for capturing long-range dependencies with dilated convolutions.
-   `TransformerModel`: A simplified Transformer encoder-based model for sequence processing, incorporating positional encoding.

### `anomaly_detection.py`
-   `Autoencoder`: A simple autoencoder for reconstruction error-based anomaly detection.
-   `VAE`: A Variational Autoencoder for probabilistic anomaly detection.
-   `AnomalyDetector`: A generic class to train and evaluate autoencoder/VAE models for anomaly detection, including threshold setting.
-   `ForecastingAnomalyDetector`: Detects anomalies based on prediction errors from a forecasting model.

### `uncertainty.py`
-   `MCDropoutModel`: Wraps a base model to enable Monte Carlo Dropout for uncertainty estimation during inference.
-   `ProbabilisticForecaster`: Utilizes `MCDropoutModel` to predict with uncertainty and provides methods to set probability-based alert thresholds.

### `visualization.py`
-   `plot_time_series_with_anomalies`: Generates plots of time series data with detected anomalies highlighted.
-   `plot_prediction_confidence_intervals`: Visualizes model predictions along with their confidence intervals.

## Future Work
-   Integration with real-time data streams for operational monitoring.
-   Exploration of more advanced deep learning architectures and anomaly detection techniques.
-   Comprehensive hyperparameter tuning and model optimization.
-   Deployment as a service for Digital Ocean Twins integration.



