import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

# === CONFIGURATION ===
data_dir = "../data"  # Path to your substation CSVs
output_csv = "../etc/substation_profiles.csv"

# === OUTPUT HOLDER ===
all_profiles = []

# === MAIN LOOP ===
csv_files = glob(os.path.join(data_dir, "*.csv"))

for file in tqdm(csv_files, desc="Processing substations"):
    try:
        # Read and parse datetime
        df = pd.read_csv(file, parse_dates=["data_collection_log_timestamp"])
        df = df.sort_values("data_collection_log_timestamp")

        # Extract daily groups
        df["date"] = df["data_collection_log_timestamp"].dt.date
        daily_groups = df.groupby("date")

        daily_profiles = []
        for _, group in daily_groups:
            if len(group) >= 48:
                # Take the first 48 values (first full day)
                energy = group.iloc[:48]["total_consumption_active_import"].values
                if np.any(np.isnan(energy)) or np.sum(energy) == 0:
                    continue  # skip incomplete or all-zero days
                daily_profiles.append(energy)

        if len(daily_profiles) == 0:
            substation_id = os.path.basename(file).replace(".csv", "")[8:]  # just for logging
            print(f"Skipping substation {substation_id} because no valid days")
            continue  # skip if no valid days

        daily_profiles = np.array(daily_profiles)  # shape: [n_days, 48]

        # === RAW MEAN PROFILE ===
        mean_profile_raw = daily_profiles.mean(axis=0)

        # === NORMALIZED MEAN PROFILE ===
        normalized_profiles = daily_profiles / daily_profiles.sum(axis=1, keepdims=True)
        mean_profile_normalized = normalized_profiles.mean(axis=0)

        # === COMBINE BOTH ===
        combined_profile = np.concatenate([mean_profile_raw, mean_profile_normalized])

        # === SAVE ROW ===
        substation_id = os.path.basename(file).replace(".csv", "")[8:]  # strip "2024-12-"
        all_profiles.append([substation_id] + combined_profile.tolist())

    except Exception as e:
        print(f"Error processing {file}: {e}")

# === SAVE TO CSV ===
columns = (
    ["substation_id"]
    + [f"raw_{i}" for i in range(48)]
    + [f"normalized_{i}" for i in range(48)]
)
profile_df = pd.DataFrame(all_profiles, columns=columns)

# Create output directory if needed
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# Save the final DataFrame
profile_df.to_csv(output_csv, index=False)
print(f"Saved substation profiles to: {output_csv}")
