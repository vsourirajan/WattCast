import re
import numpy as np

# Define log file path
log_file_path = '../etc/experiment_logs/lstm_embeddings_logs/lstm_training_embeddings.log'  # Change this to your log file path
output_file = 'lstm_embeddings_results.txt'

# Regular expression to match metrics - now matches each metric on its own line
metrics_pattern = re.compile(
    r"--- Metrics for Feeder (.+?) ---\n"
    r".*?RMSE: ([0-9.]+|nan)\n"
    r".*?MAE: ([0-9.]+|nan)\n"
    r".*?MSE: ([0-9.]+|nan)\n"
    r".*?MAPE: ([0-9.]+|nan)%\n"
    r".*?R²: ([0-9.]+|nan)",
    re.DOTALL  # Use DOTALL to match across multiple lines
)

# Initialize data storage
metrics_data = {"RMSE": [], "MAE": [], "MSE": [], "MAPE": [], "R2": []}

# Extract metrics
with open(log_file_path, 'r') as file:
    log_content = file.read()
    matches = metrics_pattern.finditer(log_content)
    for match in matches:
        _, rmse, mae, mse, mape, r2 = match.groups()
        if rmse != 'nan':
            metrics_data["RMSE"].append(float(rmse))
        if mae != 'nan':
            metrics_data["MAE"].append(float(mae))
        if mse != 'nan':
            metrics_data["MSE"].append(float(mse))
        if mape != 'nan':
            metrics_data["MAPE"].append(float(mape))
        if r2 != 'nan':
            metrics_data["R2"].append(float(r2))

# Function to calculate statistics
def calculate_statistics(values):
    values = np.array(values)
    if len(values) == 0:
        return None, None, None, None, None, None, None, None, 0
    mean = np.mean(values)
    std = np.std(values)
    median = np.median(values)
    q1, q3 = np.percentile(values, 25), np.percentile(values, 75)
    iqr = q3 - q1
    minimum = np.min(values)
    maximum = np.max(values)
    outliers = np.sum((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr))

    return mean, std, median, iqr, q1, q3, minimum, maximum, outliers

# Calculate statistics for each metric
with open(output_file, 'w') as out_file:
    for metric, values in metrics_data.items():
        mean, std, median, iqr, q1, q3, min_val, max_val, outliers = calculate_statistics(values)
        out_file.write(f"Metric: {metric}\n")
        out_file.write(f"Mean: {mean if mean is not None else 'N/A'}\n")
        out_file.write(f"Std: {std if std is not None else 'N/A'}\n")
        out_file.write(f"Median: {median if median is not None else 'N/A'}\n")
        out_file.write(f"IQR: {iqr if iqr is not None else 'N/A'} (Q1: {q1 if q1 is not None else 'N/A'}, Q3: {q3 if q3 is not None else 'N/A'})\n")
        out_file.write(f"Min: {min_val if min_val is not None else 'N/A'}\n")
        out_file.write(f"Max: {max_val if max_val is not None else 'N/A'}\n")
        out_file.write(f"Outliers: {outliers}\n")
        out_file.write(f"Number of samples: {len(values)}\n")
        out_file.write("\n")

print(f"Metrics analysis written to {output_file}")
