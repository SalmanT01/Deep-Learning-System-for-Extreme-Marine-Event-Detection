
import matplotlib.pyplot as plt
#from main import plot_time_series_with_anomalies, plot_prediction_confidence_intervals
import numpy as np

def plot_time_series_with_anomalies(time_series, anomalies, title="Time Series with Detected Anomalies", feature_names=None):
    """
    Plots a time series and highlights detected anomalies.

    Args:
        time_series (np.ndarray): The time series data (2D array: timesteps x features).
        anomalies (np.ndarray): A boolean array indicating anomalies (1D array: timesteps).
        title (str): Title of the plot.
        feature_names (list): List of names for each feature.
    """
    num_features = time_series.shape[1]
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(num_features)]

    plt.figure(figsize=(15, 5 * num_features))

    for i in range(num_features):
        plt.subplot(num_features, 1, i + 1)
        plt.plot(time_series[:, i], label=feature_names[i])
        
        # Highlight anomalies
        anomaly_indices = np.where(anomalies)[0]
        plt.scatter(anomaly_indices, time_series[anomaly_indices, i], color='red', s=50, zorder=5, label='Anomaly')
        
        plt.title(f'{title} - {feature_names[i]}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('time_series_anomalies.png')
    plt.show()

def plot_prediction_confidence_intervals(actual_values, mean_predictions, std_predictions, title="Model Prediction with Confidence Intervals", feature_names=None, confidence_level=0.95):
    """
    Plots model predictions with confidence intervals.

    Args:
        actual_values (np.ndarray): Actual observed values (2D array: timesteps x features).
        mean_predictions (np.ndarray): Mean predictions from the model (2D array: timesteps x features).
        std_predictions (np.ndarray): Standard deviations of predictions (2D array: timesteps x features).
        title (str): Title of the plot.
        feature_names (list): List of names for each feature.
        confidence_level (float): Confidence level for the prediction interval (e.g., 0.95 for 95%).
    """
    from scipy.stats import norm

    num_features = actual_values.shape[1]
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(num_features)]

    z_score = norm.ppf(1 - (1 - confidence_level) / 2)

    plt.figure(figsize=(15, 5 * num_features))

    for i in range(num_features):
        plt.subplot(num_features, 1, i + 1)
        plt.plot(actual_values[:, i], label='Actual Values', color='blue')
        plt.plot(mean_predictions[:, i], label='Mean Predictions', color='green', linestyle='--')

        upper_bound = mean_predictions[:, i] + z_score * std_predictions[:, i]
        lower_bound = mean_predictions[:, i] - z_score * std_predictions[:, i]

        plt.fill_between(range(len(mean_predictions)), lower_bound, upper_bound, color='green', alpha=0.2, label=f'{int(confidence_level*100)}% Confidence Interval')
        
        plt.title(f'{title} - {feature_names[i]}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('prediction_confidence_intervals.png')
    plt.show()

# visualization.py

def visualize_results(data_loader_obj, y_test, mean_preds, std_preds, anomalies_detected, features):
    """
    Generates visualizations for time series data, anomalies, and prediction confidence intervals.
    """
    print("\n--- Generating Visualizations ---")

    full_data_unscaled = data_loader_obj.inverse_transform(data_loader_obj.data)
    full_X, full_y = data_loader_obj._create_sequences(data_loader_obj.data)
    train_size_sequences = int(len(full_X) * data_loader_obj.train_split)

    full_anomalies_array = np.zeros(len(full_X), dtype=bool)
    full_anomalies_array[train_size_sequences : train_size_sequences + len(anomalies_detected)] = anomalies_detected

    actual_values_unscaled = data_loader_obj.inverse_transform(y_test.cpu().numpy())
    mean_preds_unscaled = data_loader_obj.inverse_transform(mean_preds)

    plot_time_series_with_anomalies(
        actual_values_unscaled,
        anomalies_detected,
        title="Test Set: Time Series with Detected Anomalies",
        feature_names=features
    )

    plot_prediction_confidence_intervals(
        actual_values_unscaled,
        mean_preds_unscaled,
        std_preds,
        title="Test Set: Model Prediction with Confidence Intervals",
        feature_names=features
    )

    print("Demonstration complete. Check 'time_series_anomalies.png' and 'prediction_confidence_intervals.png' for visualizations.")


if __name__ == '__main__':
    # Dummy data for testing
    time_steps = 100
    num_features = 3

    # Dummy time series data
    dummy_time_series = np.random.randn(time_steps, num_features).cumsum(axis=0)

    # Dummy anomalies (e.g., 5 random anomalies)
    dummy_anomalies = np.zeros(time_steps, dtype=bool)
    anomaly_indices = np.random.choice(time_steps, 5, replace=False)
    dummy_anomalies[anomaly_indices] = True

    # Plot time series with anomalies
    plot_time_series_with_anomalies(dummy_time_series, dummy_anomalies, feature_names=["Hs", "Tp", "SLP"])

    # Dummy prediction data
    dummy_actual_values = np.random.randn(time_steps, num_features).cumsum(axis=0)
    dummy_mean_predictions = dummy_actual_values + np.random.randn(time_steps, num_features) * 0.5
    dummy_std_predictions = np.random.rand(time_steps, num_features) * 0.5 + 0.1 # Ensure std is positive

    # Plot prediction confidence intervals
    plot_prediction_confidence_intervals(dummy_actual_values, dummy_mean_predictions, dummy_std_predictions, feature_names=["Hs", "Tp", "SLP"])


