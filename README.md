# Deep Learning for Extreme Marine Events Detection and Forecasting

## Project Overview
This project implements a deep learning-based system for detecting and forecasting "Extreme Marine Events", such as rogue waves and storm surges, using metocean datasets. It leverages state-of-the-art deep learning methods for:

- Anomaly detection in time series data
- Time series classification to identify patterns preceding extreme events
- Probabilistic forecasting to quantify uncertainty in event prediction

---

## ✨ Features

**Data Handling**  
- Loads and preprocesses NetCDF-formatted metocean data "MetO-NWS-WAV-RAN_1751155418549" download from Copernicus Marine Data Store | Copernicus Marine Service
data.marine.copernicus.eu (with variables e.g., significant wave height (Hs), peak period (Tp), sea level pressure)

**Sequence Modeling**  
- Implements deep learning architectures for time series modeling, including:
  - LSTM (Long Short-Term Memory)
  - GRU (Gated Recurrent Unit)
  - TCN (Temporal Convolutional Network)
  - Transformer-based models

**Anomaly Detection**  
- Supports multiple frameworks:
  - Autoencoder-based reconstruction error detection
  - Variational Autoencoders (VAE)
  - Forecasting-based detection using model prediction errors

**Uncertainty Quantification**  
- Utilizes Monte Carlo Dropout to estimate prediction uncertainty

**Visualization**  
- Generates time series plots with detected anomalies and confidence intervals for predictions

**Reproducibility**  
- Ensures reproducible experiments through controlled random seed settings

---

**Clone the repository and Run Commands**

```bash
git clone https://github.com/SalmanT01/Deep-Learning-System-for-Extreme-Marine-Event-Detection.git 
cd Deep-Learning-System-for-Extreme-Marine-Event-Detection
```

```bash
python3 -m venv venv
.venv/Scripts/activate
pip install -r requirements.txt
python .\main.py
```
