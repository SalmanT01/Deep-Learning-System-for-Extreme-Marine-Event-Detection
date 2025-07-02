
import xarray as xr
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

class MarineDataLoader:
    def __init__(self, data_path, features=None, target=None, seq_length=10, train_split=0.8):
        self.data_path = data_path
        self.user_features = features # Store user-provided features
        self.target = target
        self.seq_length = seq_length
        self.train_split = train_split
        self.scaler = MinMaxScaler()
        self.data = self._load_and_preprocess_data()

    def _load_and_preprocess_data(self):
        # Load NetCDF data using xarray
        ds = xr.open_dataset(self.data_path)
        
        if self.user_features is None:
            # Dynamically select numerical variables as features
            self.features = []
            for var_name, var in ds.variables.items():
                # Exclude coordinate variables and non-numeric data
                if var_name not in ds.dims and np.issubdtype(var.dtype, np.number):
                    # Check if the variable has a time dimension, or is a single value
                    # For simplicity, let's assume we want variables that change over time
                    if 'time' in var.dims:
                        self.features.append(var_name)
            if not self.features:
                raise ValueError("No suitable numerical features found in the dataset. Please specify features manually.")
        else:
            self.features = self.user_features

        # Ensure target is in features if specified
        if self.target and self.target not in self.features:
            if self.user_features is None: # Only add if features were auto-detected
                self.features.append(self.target)
            else:
                raise ValueError(f"Target feature \'{self.target}\' not found in specified features.")

        # Select relevant data and flatten spatial dimensions if they exist
        selected_data = ds[self.features]
        
        # Reshape data to (time, features) by averaging or selecting a point if spatial dims exist
        # For simplicity, if spatial dimensions exist, we'll take the mean across them.
        # A more robust solution might involve selecting a specific point or handling multiple points.
        if 'x' in selected_data.dims and 'y' in selected_data.dims:
            df = selected_data.mean(dim=['x', 'y']).to_dataframe()
        elif 'longitude' in selected_data.dims and 'latitude' in selected_data.dims:
            df = selected_data.mean(dim=['longitude', 'latitude']).to_dataframe()
        else:
            df = selected_data.to_dataframe()

        df = df.dropna()

        # Normalize data
        scaled_data = self.scaler.fit_transform(df[self.features].values)
        return scaled_data

    def _create_sequences(self, data):
        xs, ys = [], []
        for i in range(len(data) - self.seq_length):
            x = data[i:(i + self.seq_length)]
            # For forecasting, y should be the next time step's features
            # If self.target is None, it means we are predicting all features
            if self.target:
                # If a specific target feature is defined, predict only that
                target_idx = self.features.index(self.target)
                y = data[i + self.seq_length, target_idx]
            else:
                # Predict all features at the next time step
                y = data[i + self.seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def get_train_test_data(self):
        X, y = self._create_sequences(self.data)
        
        # Split data into training and testing sets
        train_size = int(len(X) * self.train_split)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        return X_train, y_train, X_test, y_test

    def inverse_transform(self, data):
        # Create a dummy array with zeros for features not being inverse transformed
        # This is a workaround because MinMaxScaler expects the same number of features as it was fitted on.
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # If the input data has fewer columns than the original features, pad it.
        # This assumes the data being inverse transformed corresponds to the first `data.shape[1]` features.
        if data.shape[1] < len(self.features):
            padding = np.zeros((data.shape[0], len(self.features) - data.shape[1]))
            data_padded = np.hstack((data, padding))
            return self.scaler.inverse_transform(data_padded)[:, :data.shape[1]]
        else:
            return self.scaler.inverse_transform(data)

if __name__ == '__main__':
    # Example Usage (requires a dummy NetCDF file)
    # Create a dummy NetCDF file for testing
    import netCDF4
    import os

    dummy_data_path = 'extreme_marine_events/data/dummy_metocean.nc'
    
    # Now test the DataLoader with dynamic feature detection
    print("\n--- Testing DataLoader with dynamic feature detection ---")
    data_loader_dynamic = MarineDataLoader(
        data_path=dummy_data_path,
        seq_length=20
    )
    print(f"Dynamically detected features: {data_loader_dynamic.features}")
    X_train_dyn, y_train_dyn, X_test_dyn, y_test_dyn = data_loader_dynamic.get_train_test_data()
    print(f"X_train_dyn shape: {X_train_dyn.shape}")
    print(f"y_train_dyn shape: {y_train_dyn.shape}")

    # Test the DataLoader with specified features
    print("\n--- Testing DataLoader with specified features ---")
    data_loader_specified = MarineDataLoader(
        data_path=dummy_data_path,
        seq_length=20
    )
    print(f"Specified features: {data_loader_specified.features}")
    X_train_spec, y_train_spec, X_test_spec, y_test_spec = data_loader_specified.get_train_test_data()
    print(f"X_train_spec shape: {X_train_spec.shape}")
    print(f"y_train_spec shape: {y_train_spec.shape}")

    # Test inverse transform
    sample_data = np.random.rand(1, len(data_loader_dynamic.features))
    original_data = data_loader_dynamic.inverse_transform(sample_data)
    print(f"Sample scaled data: {sample_data.flatten()}")
    print(f"Sample original data: {original_data.flatten()}")



