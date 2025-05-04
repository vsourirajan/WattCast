import re
import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
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
                    if not any(np.isnan(v) for v in metrics.values()):
                        row = {"Feeder": feeder, "Model": model}
                        row.update({k: float(v) for k, v in metrics.items()})
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
            values = group[metric].values
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = group[(group[metric] < lower) | (group[metric] > upper)]

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
                "iqr": (q1, q3),
                "num_outliers": len(outliers),
                "total": len(group)
            }
        summary_stats[model] = model_stats

    return summary_stats, pd.DataFrame(outlier_rows)

def print_statistics(summary_stats):
    for model, metrics in summary_stats.items():
        print(f"\nModel: {model}")
        for metric, vals in metrics.items():
            q1, q3 = vals['iqr']
            print(f"  {metric}: Mean = {vals['mean']:.2f}, Std = {vals['std']:.2f}, "
                  f"Median = {vals['median']:.2f}, IQR = [{q1:.2f}, {q3:.2f}], "
                  f"Outliers = {vals['num_outliers']}/{vals['total']}")

# def plot_distributions(df, output_dir="plots"):
#     os.makedirs(output_dir, exist_ok=True)
#     for metric in ["RMSE", "MAE", "MAPE", "R2"]:
#         plt.figure(figsize=(12, 6))
#         sns.boxplot(data=df, x="Model", y=metric)
#         plt.title(f"{metric} Distribution by Model")
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         plt.savefig(f"{output_dir}/{metric}_boxplot.png")
#         plt.close()

def main():
    log_path = "baseline_results.log"  # Replace with actual path
    df = parse_log_file(log_path)
    print(f"Parsed {len(df)} rows from log.")

    summary_stats, outlier_df = compute_statistics(df)

    print_statistics(summary_stats)

    df.to_csv("all_metrics_cleaned.csv", index=False)
    outlier_df.to_csv("flagged_outliers.csv", index=False)
    # plot_distributions(df)

    print("\nSaved:")
    print("  - Cleaned metrics: all_metrics_cleaned.csv")
    print("  - Outliers: flagged_outliers.csv")
    # print("  - Plots: in 'plots/' folder")

if __name__ == "__main__":
    main()
