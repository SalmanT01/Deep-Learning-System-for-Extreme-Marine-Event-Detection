import xarray as xr
import pandas as pd
import numpy as np

# -------------------------------
# Step 1: Set file paths
# -------------------------------
input_nc_file = 'D:\Deep Learning System for Extreme Marine Event Detection\extreme_marine_events\data\cmems_mod_glo_wav_anfc_0.083deg_PT3H-i_1751151345120.nc'    # Replace with your NetCDF file path
output_csv_file = "D:\Deep Learning System for Extreme Marine Event Detection\extreme_marine_events\data\cmems_mod_glo_wav_anfc_0.083deg_PT3H-i_1751151345120.csv"  # Desired output CSV file name

# -------------------------------
# Step 2: Open the NetCDF dataset
# -------------------------------
ds = xr.open_dataset(input_nc_file)

# -------------------------------
# Step 3: Flatten and convert to DataFrame
# -------------------------------
# Automatically converts multi-dimensional data to a tidy tabular format
df = ds.to_dataframe().reset_index()

# -------------------------------
# Step 4: Save to CSV
# -------------------------------
df.to_csv(output_csv_file, index=False)

print(f"Conversion complete! CSV saved as: {output_csv_file}")
