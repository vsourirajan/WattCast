import pandas as pd
import numpy as np

def create_dummy_dataset():
    df = pd.read_parquet("2024-12.parquet", filters=[("secondary_substation_unique_id", "==", "NGED-110191")])
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nNumber of rows in the dataset:")
    print(len(df))
    print("\nNumber of unique feeders at substation NGED-110191:")
    print(df["lv_feeder_unique_id"].nunique())

    df.to_csv("NGED-110191.csv")

if __name__ == "__main__":
    create_dummy_dataset()