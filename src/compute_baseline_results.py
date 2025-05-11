import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def parse_log_file(log_path):
    pattern = re.compile(
        r"- Feeder (?P<feeder>SSEN-\d+) - (?P<model>[\w\s\(\)-]+) - (?P<metrics>{.*})"
    )
    rows = []

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                feeder = match.group("feeder")
                model = match.group("model").strip()
                metrics_str = match.group("metrics")
                try:
                    metrics = eval(metrics_str, {"np": np, "__builtins__": {}})
                    # Only add row if none of the metrics are NaN
                    if all(not pd.isna(v) and v != 'nan' for v in metrics.values()):
                        row = {"Feeder": feeder, "Model": model}
                        # Convert values to float, explicitly handling 'nan' strings
                        row.update({k: float(v) if v != 'nan' else np.nan for k, v in metrics.items()})
                        # Only append if we have valid numerical values
                        if not any(pd.isna(v) for v in row.values()):
                            rows.append(row)
                except Exception as e:
                    print(f"Skipping line due to error: {e}")
    return pd.DataFrame(rows)

def compute_statistics(df):
    summary_stats = {}
    outlier_rows = []

    for model, group in df.groupby("Model"):
        model_stats = {}
        for metric in ["MSE", "RMSE", "MAE", "MAPE", "R2"]:
            # Get non-NaN values only
            values = group[metric].dropna().values
            if len(values) == 0:
                model_stats[metric] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "median": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "iqr": (np.nan, np.nan),
                    "num_outliers": 0,
                    "total": 0
                }
                continue

            # Calculate statistics only on non-NaN values
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            
            # Find outliers in non-NaN values
            outliers = group[
                (group[metric].notna()) & 
                ((group[metric] < lower) | (group[metric] > upper))
            ]

            # Add outlier rows
            for _, row in outliers.iterrows():
                outlier_rows.append({
                    "Feeder": row["Feeder"],
                    "Model": model,
                    "Metric": metric,
                    "Value": row[metric]
                })

            model_stats[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "median": np.median(values),
                "min": np.min(values),
                "max": np.max(values),
                "iqr": (q1, q3),
                "num_outliers": len(outliers),
                "total": len(values)  # Count of non-NaN values
            }
        summary_stats[model] = model_stats

    return summary_stats, pd.DataFrame(outlier_rows)

def print_statistics(summary_stats):
    for model, metrics in summary_stats.items():
        print(f"\nModel: {model}")
        for metric, vals in metrics.items():
            q1, q3 = vals['iqr']
            print(f"  {metric}:")
            if pd.isna(vals['mean']):
                print(f"    No valid data points")
                continue
            print(f"    Mean = {vals['mean']:.2f}, Std = {vals['std']:.2f}")
            print(f"    Median = {vals['median']:.2f}, IQR = [{q1:.2f}, {q3:.2f}]")
            print(f"    Min = {vals['min']:.2f}, Max = {vals['max']:.2f}")
            print(f"    Outliers = {vals['num_outliers']}/{vals['total']} (valid samples)")

def plot_distributions(df, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    for metric in ["RMSE", "MAE", "MAPE", "R2"]:
        # Filter out NaN values before plotting
        plot_data = df.dropna(subset=[metric])
        if len(plot_data) == 0:
            print(f"No valid data points for {metric}, skipping plot")
            continue
            
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=plot_data, x="Model", y=metric)
        plt.title(f"{metric} Distribution by Model (excluding NaN values)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{metric}_boxplot.png")
        plt.close()

def find_worst_performers(df, n=10):
    worst_performers = {}
    
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        model_worst = {}
        
        # For each metric, drop NaN values before finding worst performers
        for metric in ['R2', 'RMSE', 'MAE']:
            valid_data = model_data.dropna(subset=[metric])
            if len(valid_data) == 0:
                continue
                
            if metric == 'R2':
                # For R2, lower values are worse
                worst = valid_data.nsmallest(n, metric)[['Feeder', metric]]
            else:
                # For RMSE and MAE, higher values are worse
                worst = valid_data.nlargest(n, metric)[['Feeder', metric]]
            
            model_worst[metric] = worst
        
        if model_worst:  # Only add if we have any valid metrics
            worst_performers[model] = model_worst
    
    return worst_performers

def print_worst_performers(worst_performers):
    print("\nWorst Performing Feeders by Model and Metric:")
    print("=" * 80)
    
    # Create output string
    output_lines = []
    output_lines.append("Worst Performing Feeders by Model and Metric:")
    output_lines.append("=" * 80)
    
    for model, metrics in worst_performers.items():
        print(f"\nModel: {model}")
        print("-" * 40)
        output_lines.append(f"\nModel: {model}")
        output_lines.append("-" * 40)
        
        for metric, data in metrics.items():
            print(f"\n{metric} (worst {len(data)} feeders):")
            output_lines.append(f"\n{metric} (worst {len(data)} feeders):")
            for _, row in data.iterrows():
                line = f"  Feeder: {row['Feeder']}, Value: {row[metric]:.4f}"
                print(line)
                output_lines.append(line)
    
    # Save to file
    os.makedirs("../etc", exist_ok=True)
    with open("../etc/worst_performers.txt", "w") as f:
        f.write("\n".join(output_lines))

def main():
    log_path = "../etc/experiment_logs/baseline_results.log"  # Replace with actual path
    df = parse_log_file(log_path)
    print(f"Parsed {len(df)} rows from log.")

    summary_stats, outlier_df = compute_statistics(df)
    worst_performers = find_worst_performers(df)

    print_statistics(summary_stats)
    print_worst_performers(worst_performers)

    df.to_csv("all_metrics_cleaned.csv", index=False)
    outlier_df.to_csv("flagged_outliers.csv", index=False)
    plot_distributions(df)

    print("\nSaved:")
    print("  - Cleaned metrics: all_metrics_cleaned.csv")
    print("  - Outliers: flagged_outliers.csv")
    print("  - Plots: in 'plots/' folder")
    print("  - Worst performers: ../etc/worst_performers.txt")

if __name__ == "__main__":
    main()
