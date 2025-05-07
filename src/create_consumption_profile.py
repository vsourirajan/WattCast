import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

# === CONFIGURATION ===
data_dir = "../data"  # Path to your substation CSVs
output_csv = "../etc/substation_profiles.csv"
alpha = 0.0001  # Scaling factor for log transformation

# === OUTPUT HOLDER ===
all_profiles = []

# === MAIN LOOP ===
csv_files = glob(os.path.join(data_dir, "*.csv"))

for file in tqdm(csv_files, desc="Processing substations"):
    try:
        # Read and parse datetime
        df = pd.read_csv(file, parse_dates=["data_collection_log_timestamp"])
        df = df.sort_values("data_collection_log_timestamp")

        # Aggregate consumption across all feeders by timestamp
        df_grouped = df.groupby("data_collection_log_timestamp")["total_consumption_active_import"].sum().reset_index()

        # Extract first 48 values for the first full day
        if len(df_grouped) < 48:
            print(f"Skipping {file} - less than 48 timestamps")
            continue

        first_48_values = df_grouped.iloc[:48]["total_consumption_active_import"].values

        # Check if all values are zero
        if np.sum(first_48_values) == 0:
            print(f"Skipping {file} - all-zero consumption")
            continue

        normalized_profile = first_48_values / np.max(first_48_values)

        # === LOG-TRANSFORMED PROFILE WITH MAGNITUDE ===
        magnitude_preserved_profile = np.log1p(alpha * first_48_values)  # Scaled log transformation

        # print("Sample transformed profile:", magnitude_preserved_profile)
        # print("Max value:", np.max(magnitude_preserved_profile))
        # print("Min value:", np.min(magnitude_preserved_profile))


        # Save profile

        substation_id = os.path.basename(file).replace(".csv", "")[8:]  # Extract substation ID
        all_profiles.append([substation_id] + magnitude_preserved_profile.tolist())

    except Exception as e:
        print(f"Error processing {file}: {e}")

# === SAVE TO CSV ===
columns = ["substation_id"] + [f"magnitude_log_{i}" for i in range(48)]
profile_df = pd.DataFrame(all_profiles, columns=columns)

# Create output directory if needed
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# Save the final DataFrame
profile_df.to_csv(output_csv, index=False)
print(f"Saved substation profiles to: {output_csv}")