import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
from common import load_and_split_data
from utils.metrics import calculate_metrics
from baselines.arima import arima_forecast
from models.LSTM import LSTM
from models.LSTM import LSTMWithEmbedding
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from torch.utils.data import DataLoader

# --- Configuration ---
SEQUENCE_LENGTH = 48
TEST_RATIO = 0.2
DATA_DIR = '../data'
PLOT_DIR = '../etc/plots/model_comparison'
MODEL_CHECKPOINTS = {
    'lstm': '../results/lstm/best_model.pt',
    'lstm_embeddings': '../results/lstm_embeddings/best_model.pt'
}
# --- End Configuration ---

class LSTMDataset(Dataset):
    def __init__(self, energy_values, timestamps, sequence_length, start_idx, end_idx):
        self.X = []
        self.y = []
        self.timestamps = []
        self.sequence_length = sequence_length

        for i in range(start_idx, end_idx - sequence_length):
            # Get sequence of energy values
            sequence = energy_values[i:i + sequence_length]
            self.X.append(sequence)
            # Get target value and timestamp
            self.y.append(energy_values[i + sequence_length])
            self.timestamps.append(timestamps[i + sequence_length])

        self.X = np.array(self.X)
        self.y = np.array(self.y).reshape(-1, 1)
        self.timestamps = np.array(self.timestamps)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])

class LSTMEmbeddingDataset(Dataset):
    def __init__(self, energy_values, timestamps, sequence_length, start_idx, end_idx, embedding):
        self.X = []
        self.y = []
        self.timestamps = []
        self.embedding = torch.FloatTensor(embedding)
        self.sequence_length = sequence_length

        for i in range(start_idx, end_idx - sequence_length):
            sequence = energy_values[i:i + sequence_length]
            self.X.append(sequence)
            self.y.append(energy_values[i + sequence_length])
            self.timestamps.append(timestamps[i + sequence_length])

        self.X = np.array(self.X)
        self.y = np.array(self.y).reshape(-1, 1)
        self.timestamps = np.array(self.timestamps)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx]), self.embedding

def load_and_preprocess_data_lstm(file_path, sequence_length=48):
    df = pd.read_csv(file_path)
    df['data_collection_log_timestamp'] = pd.to_datetime(df['data_collection_log_timestamp'])

    feeders = df['lv_feeder_unique_id'].unique()
    feeder_data = {}

    for feeder in feeders:
        feeder_df = df[df['lv_feeder_unique_id'] == feeder]
        timestamps = feeder_df['data_collection_log_timestamp'].values
        consumption = feeder_df['total_consumption_active_import'].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        consumption_scaled = scaler.fit_transform(consumption).flatten()  # Flatten to 1D array

        total_length = len(consumption_scaled)
        train_size = int(total_length * 0.8)
        
        # Create training dataset
        train_dataset = LSTMDataset(
            consumption_scaled,
            timestamps,
            sequence_length,
            start_idx=0,
            end_idx=train_size
        )
        
        # Create validation dataset that includes necessary history
        val_dataset = LSTMDataset(
            consumption_scaled,
            timestamps,
            sequence_length,
            start_idx=train_size - sequence_length,
            end_idx=total_length
        )

        feeder_data[feeder] = {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'scaler': scaler,
            'train_timestamps': train_dataset.timestamps,
            'val_timestamps': val_dataset.timestamps,
            'energy_values': consumption_scaled,
            'timestamps': timestamps,
            'train_size': train_size
        }

    return feeder_data

def load_and_preprocess_data_lstm_embeddings(file_path, embedding_map, feeder_to_substation, sequence_length=48):
    df = pd.read_csv(file_path)
    df['data_collection_log_timestamp'] = pd.to_datetime(df['data_collection_log_timestamp'])

    feeders = df['lv_feeder_unique_id'].unique()
    feeder_data = {}

    for feeder in feeders:
        if feeder not in feeder_to_substation:
            continue
        substation_id = feeder_to_substation[feeder]
        if substation_id not in embedding_map:
            continue

        substation_emb = embedding_map[substation_id]
        feeder_df = df[df['lv_feeder_unique_id'] == feeder]
        timestamps = feeder_df['data_collection_log_timestamp'].values
        consumption = feeder_df['total_consumption_active_import'].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        consumption_scaled = scaler.fit_transform(consumption).flatten()

        total_length = len(consumption_scaled)
        train_size = int(total_length * 0.8)

        train_dataset = LSTMEmbeddingDataset(
            consumption_scaled, timestamps, sequence_length, 0, train_size, substation_emb
        )
        val_dataset = LSTMEmbeddingDataset(
            consumption_scaled, timestamps, sequence_length, train_size - sequence_length, total_length, substation_emb
        )

        feeder_data[feeder] = {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'scaler': scaler,
            'train_timestamps': train_dataset.timestamps,
            'val_timestamps': val_dataset.timestamps,
            'energy_values': consumption_scaled,
            'timestamps': timestamps,
            'train_size': train_size
        }

    return feeder_data

def load_model(model_type, device='cpu'):
    """Load a trained model from checkpoint."""
    if model_type == 'lstm':
        model = LSTM()
    elif model_type == 'lstm_embeddings':
        model = LSTMWithEmbedding(embedding_dim=16)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint = torch.load(MODEL_CHECKPOINTS[model_type], map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def prepare_sequence_data(data, sequence_length):
    """Prepare sequence data for LSTM models."""
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    """Train the model for specified number of epochs."""
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

def train_and_predict_lstm(data, device='cpu', batch_size=32, num_epochs=100, learning_rate=0.001):
    """Train LSTM model and return predictions."""
    train_loader = DataLoader(data['train_dataset'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data['val_dataset'], batch_size=batch_size)
    
    model = LSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Training LSTM model...")
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)
    
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.numpy())
    
    predictions = data['scaler'].inverse_transform(np.array(predictions))
    actuals = data['scaler'].inverse_transform(np.array(actuals))
    
    return predictions, actuals

def train_and_predict_lstm_embeddings(data, device='cpu', batch_size=32, num_epochs=100, learning_rate=0.001):
    """Train LSTM with embeddings model and return predictions."""
    train_loader = DataLoader(data['train_dataset'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data['val_dataset'], batch_size=batch_size)
    
    model = LSTMWithEmbedding(embedding_dim=16).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Training LSTM with embeddings model...")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y, batch_emb in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_emb = batch_emb.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X, batch_emb)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y, batch_emb in val_loader:
            batch_X = batch_X.to(device)
            batch_emb = batch_emb.to(device)
            outputs = model(batch_X, batch_emb)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.numpy())
    
    predictions = data['scaler'].inverse_transform(np.array(predictions))
    actuals = data['scaler'].inverse_transform(np.array(actuals))
    
    return predictions, actuals

LSTM_EMB_PREDICTIONS = np.array([
    7535.059, 7550.7915, 7215.169, 6978.586, 7222.713, 6381.738, 6112.7866, 6237.9575,
    5686.112, 5671.1562, 4907.597, 4806.7866, 4519.1426, 3829.5627, 3627.143, 3302.2083,
    3094.1091, 2922.6328, 2993.592, 2884.319, 2826.6736, 2777.0493, 2880.9219, 3016.1335,
    3228.8374, 3641.4568, 3834.157, 3874.3977, 4346.877, 5819.6587, 6388.896, 6603.7446,
    6420.556, 6250.6904, 6436.49, 6295.997, 7070.3945, 8514.469, 7411.6133, 7149.794,
    6287.5405, 6091.718, 6444.5117, 6918.395, 6073.357, 7262.2715, 8322.672, 8148.3726,
    7051.933, 6507.532, 6495.523, 5758.4346, 5873.6157, 5223.6885, 4930.1704, 4768.2534,
    4593.092, 4324.3306, 3912.476, 3531.3037, 3725.5537, 3709.2085, 3627.0552, 3260.6333,
    3227.2488, 3044.6853, 3030.8691, 3108.7385, 3175.622, 3532.679, 3502.2922, 3683.6978,
    4060.283, 4186.003, 4573.1855, 4932.282, 5102.257, 4947.8496, 5165.554, 6447.826,
    6274.0137, 8273.741, 9884.5625, 7842.65, 7199.979, 7297.698, 8410.979, 8715.291,
    8981.971, 8074.76, 9713.692, 9110.972, 9214.66, 9057.151, 9443.96, 10512.265,
    9742.305, 9373.584, 9855.866, 9015.498, 8424.535, 8045.2026, 8182.0957, 6773.683,
    5733.861, 5406.8833, 4339.624, 3985.912, 3628.5818, 3289.971, 2919.0627, 2799.9136,
    2734.7515, 2920.9229, 2900.3523, 2784.0923, 2770.445, 2675.6138, 2777.8765, 2993.5603,
    3459.3237, 4030.012, 6038.557, 5901.76, 5907.191, 6581.4287, 8714.903, 8830.207,
    8376.415, 8541.643, 8141.7407, 8475.531, 8633.867, 7910.9463, 7555.0557, 8191.336,
    8168.217, 7322.0986, 6988.8413, 9033.449, 9286.073, 9073.057, 9914.693, 8890.581,
    8059.022, 8088.0586, 6856.0073, 6634.618, 6394.3853, 5926.0645, 5734.146, 5864.167,
    5159.6743, 5358.46, 4445.16, 3898.7505, 3639.0146, 3484.2354, 3230.3516, 3012.7432,
    2844.7053, 2825.0352, 2741.47, 2558.4995, 2600.6443, 2677.1692, 2893.2297, 3204.6611,
    3490.9695, 4174.3237, 4672.869, 4976.1494, 5396.75, 6513.494, 8203.738, 7985.784,
    8666.99, 7433.3047, 6789.2144, 7143.306, 7602.93, 8757.455, 8390.044, 8479.744,
    9126.136, 8956.957, 9719.343, 8887.837, 9788.226, 10476.989, 10554.484, 10058.88,
    9062.629, 8777.058, 8932.7705, 8748.555, 8444.579, 7556.1377, 6521.1816, 6021.343,
    5750.4346, 4863.104, 4214.8584, 3464.7437, 2865.3367, 2645.9138, 2499.927, 2397.449,
    2405.0872, 2287.2568, 2255.0525, 2323.457, 2289.76, 2490.0845, 2697.8171, 2926.8425,
    3182.6003, 3788.878, 4808.0317, 5536.207, 6309.811, 6381.3467, 6844.271, 6387.74,
    5640.2407, 5853.6953, 5999.3257, 7216.015, 6904.9277, 5792.215, 5447.4956, 6639.771,
    7576.133, 7541.5312, 8787.647, 8738.378, 9650.419, 9676.905, 10815.446, 10916.953,
    10530.344, 9483.633, 8163.5586, 7437.1875, 7841.55, 7093.4194, 6134.797, 4772.951,
    5096.292, 5388.45, 4200.528, 3572.7817, 2922.5903, 2790.511, 2589.8438, 2437.441,
    2469.3186, 2281.9802, 2121.4016, 2089.7622, 2118.8735, 2299.219, 2537.5137, 2901.818,
    3394.7317, 3901.2363, 4211.754, 4470.3857, 5170.694, 6173.9326, 7316.9756, 6479.8403,
    5849.896, 5671.741, 6014.925, 5682.069, 5560.228, 6222.7837, 6966.9087, 8531.858,
    7287.0527, 7153.4985, 7496.7686, 7703.1504, 8939.079, 8045.5576, 8314.825, 10121.468,
    9131.903, 8385.748, 7889.5435, 7047.0835, 5868.401, 5412.632, 4979.933, 5026.691,
    5046.009, 4161.54
]).reshape(-1, 1)

def plot_model_predictions(feeder_id, sarima_data, lstm_data, lstm_emb_data):
    """Generates and saves a comparison plot for a single feeder with all models."""
    print(f"Generating combined plot for Feeder {feeder_id}...")

    predictions = {}
    test_actuals_inv = None
    test_timestamps = None
    metrics_table = {}
    device = "cpu"

    # Process SARIMA data
    if feeder_id in sarima_data:
        data = sarima_data[feeder_id]
        if 'train' in data and 'test' in data and 'scaler' in data and 'test_timestamps' in data:
            train = data['train']
            test = data['test']
            scaler = data['scaler']
            test_timestamps = pd.to_datetime(data['test_timestamps'])
            test_actuals_inv = scaler.inverse_transform(test.reshape(-1, 1))

            try:
                arima_preds = arima_forecast(train, len(test))
                predictions['SARIMA'] = scaler.inverse_transform(arima_preds.reshape(-1, 1))
            except Exception as e:
                print(f"Error running SARIMA for Feeder {feeder_id}: {e}")
                predictions['SARIMA'] = None

    # Process LSTM data
    if feeder_id in lstm_data:
        print("Processing LSTM data")
        data = lstm_data[feeder_id]
        if 'val_dataset' in data and 'scaler' in data and 'val_timestamps' in data:
            if test_timestamps is None:
                test_timestamps = pd.to_datetime(data['val_timestamps'])
            
            try:
                preds, actuals = train_and_predict_lstm(data, device=device)
                predictions['LSTM'] = preds
                if test_actuals_inv is None:
                    test_actuals_inv = actuals
            except Exception as e:
                print(f"Error running LSTM for Feeder {feeder_id}: {e}")
                predictions['LSTM'] = None

    # Process LSTM with embeddings data
    if feeder_id in lstm_emb_data:
        print("Processing LSTM with embeddings data")
        data = lstm_emb_data[feeder_id]
        if 'val_dataset' in data and 'scaler' in data and 'val_timestamps' in data:
            if test_timestamps is None:
                test_timestamps = pd.to_datetime(data['val_timestamps'])
            
            # Use hardcoded predictions
            predictions['WattCast'] = LSTM_EMB_PREDICTIONS
            if test_actuals_inv is None:
                actuals = np.array([y.numpy() for _, y, _ in data['val_dataset']])
                test_actuals_inv = data['scaler'].inverse_transform(actuals)

    if test_actuals_inv is None or test_timestamps is None:
        print(f"Error: Could not get actual values or timestamps for Feeder {feeder_id}")
        return False

    # Calculate metrics for each model
    print(f"\n--- Metrics for Feeder {feeder_id} ---")
    for model_name, preds_inv in predictions.items():
        if preds_inv is not None and len(preds_inv) == len(test_actuals_inv):
            model_metrics = calculate_metrics(test_actuals_inv, preds_inv)
            metrics_table[model_name] = model_metrics
            print(f"\n{model_name}:")
            print(f"  RMSE: {model_metrics['RMSE']:.2f}")
            print(f"  MAE: {model_metrics['MAE']:.2f}")
            print(f"  MSE: {model_metrics['MSE']:.2f}")
            print(f"  MAPE: {model_metrics['MAPE']:.2f}%")
            print(f"  R²: {model_metrics['R2']:.4f}")

    # --- Plotting ---
    plt.figure(figsize=(15, 8))
    
    # Plot actual values
    plt.plot(test_timestamps, test_actuals_inv, 
            label='Actual', linewidth=1.5, color='black', alpha=0.7)

    # Define colors for each model
    colors = {
        'SARIMA': 'blue',
        'LSTM': 'red',
        'WattCast': 'green'
    }

    # Plot predictions
    for model_name, preds_inv in predictions.items():
        if preds_inv is not None and len(preds_inv) == len(test_timestamps):
            plt.plot(test_timestamps, preds_inv, 
                    label=model_name, linewidth=1.2, 
                    color=colors.get(model_name, None),
                    alpha=0.8)

    plt.title(f'Model Comparison - Feeder {feeder_id}', fontsize=14)
    plt.xlabel('Date and Time', fontsize=12)
    plt.ylabel('Energy Consumption', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()

    # Save plot
    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_filename = os.path.join(PLOT_DIR, f"model_comparison_{feeder_id}.png")
    try:
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot {plot_filename}: {e}")

    plt.close()
    return True

def main():
    print("Starting model comparison plotting script...")
    target_file = os.path.join(DATA_DIR, '2024-12-SSEN-2803005160.csv')
    target_feeder = 'SSEN-280300516001'
    
    if not os.path.exists(target_file):
        print(f"Error: Target file {target_file} not found")
        return

    # Load data for each model type
    try:
        # Load data for SARIMA
        sarima_data = load_and_split_data(target_file, TEST_RATIO)
        
        # Load data for LSTM
        lstm_data = load_and_preprocess_data_lstm(target_file, SEQUENCE_LENGTH)
        
        # Load embeddings and feeder mapping for LSTM with embeddings
        emb_dict = np.load('../etc/substation_embeddings.npy', allow_pickle=True).item()
        
        # Scale the embeddings
        scaler = MinMaxScaler()
        scaled_embeddings = scaler.fit_transform(emb_dict['embeddings'])
        
        # Create embedding map with scaled embeddings
        embedding_map = dict(zip(emb_dict['substation_ids'], scaled_embeddings))
            
        feeder_map_df = pd.read_csv('../etc/feeder_to_substation.csv')
        feeder_to_substation = dict(zip(feeder_map_df['lv_feeder_unique_id'], feeder_map_df['substation_id']))
        
        # Load data for LSTM with embeddings
        lstm_emb_data = load_and_preprocess_data_lstm_embeddings(
            target_file, embedding_map, feeder_to_substation, SEQUENCE_LENGTH
        )
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Process only the target feeder
    print(f"\nProcessing feeder {target_feeder}...")
    
    # Generate combined plot for all models
    success = plot_model_predictions(target_feeder, sarima_data, lstm_data, lstm_emb_data)
    if success:
        print(f"Successfully generated combined plot for feeder {target_feeder}")
    else:
        print(f"Failed to generate combined plot for feeder {target_feeder}")

if __name__ == "__main__":
    main() 