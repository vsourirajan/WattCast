import pandas as pd
import os

def filter_and_save_substations(substation_list_file, parquet_file_path, output_dir):
    print(f"Starting substation filtering...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        with open(substation_list_file, 'r') as f:
            substation_ids = [line.strip().split()[1] for line in f if line.strip()]
        print(f"Read {len(substation_ids)} substation IDs from {substation_list_file}")
    except FileNotFoundError:
        print(f"Error: Substation list file not found at {substation_list_file}")
        return

    for sub_id in substation_ids:
        print(f"Processing substation: {sub_id}")
        try:
            sub_df = pd.read_parquet(
                parquet_file_path,
                filters=[("secondary_substation_unique_id", "==", sub_id)]
            )
            if not sub_df.empty:
                output_filename = os.path.join(output_dir, f"2024-12-{sub_id}.csv")
                sub_df.to_csv(output_filename, index=False)
                print(f"Saved data for {sub_id} to {output_filename}")
            else:
                print(f"No data found for substation {sub_id} in {parquet_file_path}")
        except Exception as e:
            print(f"Error processing substation {sub_id}: {e}")

    print("Substation filtering complete.")

def main():
    substation_list_path = 'substations_sample.txt'
    parquet_path = '../2024-12.parquet' 
    output_data_dir = '../data'
    filter_and_save_substations(substation_list_path, parquet_path, output_data_dir)

if __name__ == "__main__":
    main()
