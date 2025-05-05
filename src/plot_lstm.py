import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def plot_lstm_results(feeder_id, timestamps, actuals, predictions, save_path):
    """Plot LSTM results for a single feeder."""
    plt.figure(figsize=(12, 6))
    # Plot actual test values
    plt.plot(timestamps, actuals, label='Actual', linewidth=0.8, color='black')
    # Plot predictions
    plt.plot(timestamps, predictions, label='LSTM', linewidth=0.8)

    plt.title(f'Energy Consumption Forecasting - Feeder {feeder_id}')
    plt.xlabel('Date and Time')
    plt.ylabel('Consumption')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.gcf().autofmt_xdate()  # Rotate date labels
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf() 