import os
from common import load_and_split_data
from utils.metrics import calculate_metrics
from baselines.naive import naive_forecast
from baselines.average import mean_forecast
from baselines.arima import arima_forecast
#from baselines.boosting import boosting_forecast
from baselines.prophet import prophet_forecast

def main():
    sequence_length = 48
    test_ratio = 0.2
    data_dir = '../data'  # where all your CSVs live

    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    for file_path in files:
        print(f"\nProcessing {file_path}")
        feeder_data = load_and_split_data(file_path, sequence_length, test_ratio)

        for feeder_id, data in feeder_data.items():
            print(f"Feeder {feeder_id}:")

            train = data['train']
            test = data['test']
            scaler = data['scaler']

            # Naive
            naive_preds = naive_forecast(train, test)
            naive_preds = scaler.inverse_transform(naive_preds)
            test_actuals = scaler.inverse_transform(test)
            naive_metrics = calculate_metrics(test_actuals, naive_preds)
            print(f"Naive RMSE: {naive_metrics['RMSE']:.2f}")

            # Mean
            mean_preds = mean_forecast(train, test)
            mean_preds = scaler.inverse_transform(mean_preds)
            mean_metrics = calculate_metrics(test_actuals, mean_preds)
            print(f"Mean RMSE: {mean_metrics['RMSE']:.2f}")

            # ARIMA
            arima_preds = arima_forecast(train, len(test))
            arima_preds = scaler.inverse_transform(arima_preds)
            arima_metrics = calculate_metrics(test_actuals, arima_preds)
            print(f"ARIMA RMSE: {arima_metrics['RMSE']:.2f}")

            # Boosting
            boosting_preds = boosting_forecast(train, len(test))
            boosting_preds = scaler.inverse_transform(boosting_preds)
            boosting_metrics = calculate_metrics(test_actuals, boosting_preds)
            print(f"Boosting RMSE: {boosting_metrics['RMSE']:.2f}")

            # Prophet
            prophet_preds = prophet_forecast(train, data['train_timestamps'], len(test))
            prophet_preds = scaler.inverse_transform(prophet_preds)
            prophet_metrics = calculate_metrics(test_actuals, prophet_preds)
            print(f"Prophet RMSE: {prophet_metrics['RMSE']:.2f}")

if __name__ == "__main__":
    main()
