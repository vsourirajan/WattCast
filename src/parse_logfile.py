import re
import numpy as np
from collections import defaultdict
import json

def parse_log_file(log_path):
    # Regex to match log lines containing metrics
    pattern = re.compile(
        r"- Feeder (?P<feeder>SSEN-\d+) - (?P<model>[\w\s\(\)-]+) - (?P<metrics>{.*})"
    )

    results = defaultdict(lambda: defaultdict(list))

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                model = match.group('model').strip()
                metrics_str = match.group('metrics')
                
                try:
                    # Parse the metrics dictionary string
                    metrics = eval(metrics_str, {"np": np, "__builtins__": {}})
                    if not any(np.isnan(value) for value in metrics.values()):
                        for metric, value in metrics.items():
                            results[model][metric].append(float(value))
                except Exception as e:
                    print(f"Error parsing line: {line}\n{e}")
    
    # Compute averages
    avg_results = {
        model: {metric: np.mean(values) for metric, values in model_metrics.items()}
        for model, model_metrics in results.items()
    }

    return avg_results

def main():
    log_path = 'baseline_results.log'  # Replace with actual log file path
    avg_results = parse_log_file(log_path)

    # Print results
    for model, metrics in avg_results.items():
        print(f"\nModel: {model}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
