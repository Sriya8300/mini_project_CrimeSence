import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import os
import logging
import sys
import json
from flask import Flask, render_template, send_file, Response
from app_processing import process_video

# Set up logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# Directories and file paths
video_output_dir = 'outputs'
static_dir = 'static'
heatmap_path = os.path.join('templates', 'crime_heatmap.html')
os.makedirs(video_output_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Step 1: Preprocess historical crime data
def preprocess_data(input_file, output_file):
    if not os.path.exists(input_file):
        logger.error(f"Crime data file {input_file} not found.")
        return None
    logger.info("Loading and cleaning crime data...")
    try:
        df = pd.read_csv(input_file)
        df = df.dropna(subset=['Date', 'Crime_Type', 'Latitude', 'Longitude'])
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df[(df['Date'].dt.year >= 2019) & (df['Date'].dt.year <= 2024)]
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df['Population_Density'] = pd.to_numeric(df['Population_Density'], errors='coerce')
        df = df.dropna(subset=['Latitude', 'Longitude', 'Population_Density'])
        if df.empty:
            logger.error("No valid data after cleaning.")
            return None

        logger.info("Engineering features...")
        df['Lat_Grid'] = pd.cut(df['Latitude'], bins=5, labels=False)
        df['Lon_Grid'] = pd.cut(df['Longitude'], bins=5, labels=False)
        df['Grid_ID'] = df['Lat_Grid'].astype(str) + '_' + df['Lon_Grid'].astype(str)
        le_crime_type = LabelEncoder()
        df['Crime_Type_Code'] = le_crime_type.fit_transform(df['Crime_Type'])
        df['Urban_Characteristic_Code'] = LabelEncoder().fit_transform(df['Urban_Characteristic'])
        df['CCTV_Coverage'] = df['CCTV_Coverage'].map({'Yes': 1, 'No': 0})
        scaler = StandardScaler()
        df['Population_Density_Scaled'] = scaler.fit_transform(df[['Population_Density']])

        logger.info("Creating time-series data...")
        df['Date_Str'] = df['Date'].dt.strftime('%Y-%m-%d')
        agg_data = df.groupby(['Date_Str', 'Grid_ID']).size().reset_index(name='Crime_Count')
        date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
        grids = agg_data['Grid_ID'].unique()
        sequence_data = []
        for grid in grids:
            grid_data = agg_data[agg_data['Grid_ID'] == grid].set_index('Date_Str')['Crime_Count']
            grid_series = pd.Series(0, index=date_range.strftime('%Y-%m-%d'))
            grid_series.update(grid_data)
            np.save(os.path.join('outputs', f'crime_counts_{grid}.npy'), grid_series.values)
            sequence_data.append({'Grid_ID': grid, 'Crime_Counts_File': f'crime_counts_{grid}.npy'})

        sequence_df = pd.DataFrame(sequence_data)
        grid_features = df.groupby('Grid_ID').agg({
            'Population_Density_Scaled': 'mean',
            'Urban_Characteristic_Code': 'mean',
            'CCTV_Coverage': 'mean',
            'Crime_Type_Code': lambda x: x.mode()[0]
        }).reset_index()
        sequence_df = sequence_df.merge(grid_features, on='Grid_ID')

        sequence_df.to_csv(output_file, index=False)
        np.save(os.path.join('outputs', 'le_crime_type.npy'), le_crime_type.classes_)
        logger.info(f"Processed data saved to {output_file}")
        return sequence_df
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        return None

# Step 2: Create LSTM sequences
def create_sequences(data, seq_length=1):
    logger.info(f"Creating LSTM sequences with seq_length={seq_length}...")
    X, y, grid_ids_seq = [], [], []
    crime_counts = [np.load(os.path.join('outputs', file)) for file in data['Crime_Counts_File']]
    features = data[['Population_Density_Scaled', 'Urban_Characteristic_Code', 'CCTV_Coverage', 'Crime_Type_Code']].values
    grid_ids = data['Grid_ID'].values

    for i in range(len(crime_counts)):
        counts = crime_counts[i]
        if len(counts) < seq_length + 1:
            logger.warning(f"Skipping Grid {grid_ids[i]}: Not enough data for sequence length {seq_length}")
            continue
        for j in range(len(counts) - seq_length):
            count_seq = counts[j:j+seq_length].reshape(-1, 1)
            feature_seq = np.repeat(features[i:i+1], seq_length, axis=0)
            X.append(np.column_stack((count_seq, feature_seq)))
            y.append(counts[j+seq_length])
            grid_ids_seq.append(grid_ids[i])

    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    if len(X) == 0:
        logger.error("No valid sequences generated. Dataset may be too small or seq_length too large.")
        sys.exit(1)
    return X, y, grid_ids_seq

# Step 3: Map Grid_ID to coordinates
def grid_to_coordinates(grid_id, lat_range=(17.0, 19.0), lon_range=(77.0, 81.0), bins=5):
    try:
        lat_grid, lon_grid = map(int, grid_id.split('_'))
        lat_step = (lat_range[1] - lat_range[0]) / bins
        lon_step = (lon_range[1] - lon_range[0]) / bins
        lat = lat_range[0] + (lat_grid + 0.5) * lat_step
        lon = lon_range[0] + (lon_grid + 0.5) * lon_step
        return lat, lon
    except Exception as e:
        logger.warning(f"Invalid Grid_ID {grid_id}: {str(e)}")
        return None, None

# Step 4: Main function
def crimesense_with_cctv(crime_file, video_path, processed_file, model_output, predictions_output, heatmap_output):
    logger.info("Starting CrimeSense processing...")
    df = preprocess_data(crime_file, processed_file)
    if df is None:
        logger.error("Failed to preprocess data.")
        return None, None, None, None, None

    logger.info("Creating LSTM sequences...")
    X, y, grid_ids_seq = create_sequences(df)
    if len(X) == 0:
        logger.error("No valid sequences generated.")
        sys.exit(1)

    logger.info("Splitting data...")
    X_train, X_temp, y_train, y_temp, grid_ids_train, grid_ids_temp = train_test_split(X, y, grid_ids_seq, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test, grid_ids_val, grid_ids_test = train_test_split(X_temp, y_temp, grid_ids_temp, test_size=0.5, random_state=42)

    logger.info("Building LSTM model...")
    model = Sequential([
        LSTM(64, input_shape=(1, X_train.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    logger.info("Training model...")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)

    logger.info("Evaluating LSTM model...")
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"LSTM Test Loss (MSE): {test_loss:.4f}")

    y_pred = model.predict(X_test, verbose=0)
    predictions_df = pd.DataFrame({'Grid_ID': grid_ids_test, 'Actual': y_test, 'Predicted': y_pred.flatten()})
    predictions_df.to_csv(predictions_output, index=False)

    model.save(model_output)

    logger.info("Saving training loss plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.savefig(os.path.join('static', 'training_loss_plot.png'))
    plt.close()

    logger.info("Creating crime hotspot heatmap...")
    crime_map = folium.Map(location=[17.3850, 78.4867], zoom_start=8)
    heat_data = [[lat, lon, max(0, row['Predicted'])] for _, row in predictions_df.iterrows() if (lat := grid_to_coordinates(row['Grid_ID'])[0]) and (lon := grid_to_coordinates(row['Grid_ID'])[1])]
    HeatMap(heat_data, radius=15, blur=20).add_to(crime_map)
    crime_map.save(heatmap_output)

    summaries = process_video(video_path, video_output_dir, static_dir)
    if not summaries:
        logger.warning("No video summaries generated.")
        return model, history, predictions_df, crime_map, summaries

    return model, history, predictions_df, crime_map, summaries

# Flask Routes
@app.route('/')
def index():
    logger.info("Serving index page...")
    try:
        with open('summaries.json', 'r') as f:
            summaries = json.load(f)
        return render_template('index.html', summaries=summaries)
    except Exception as e:
        logger.error(f"Error serving index page: {str(e)}")
        return "Error loading summaries", 500

@app.route('/video')
def video():
    video_path = os.path.join('static', 'converted_test_trimmed_video.mp4')
    logger.info(f"Serving video from {video_path}")
    if not os.path.exists(video_path):
        logger.error(f"Processed video {video_path} not found.")
        return "Processed video not found.", 404
    def generate():
        with open(video_path, 'rb') as f:
            while True:
                chunk = f.read(1024 * 1024)  # Read in 1MB chunks
                if not chunk:
                    break
                yield chunk
    return Response(generate(), mimetype='video/mp4')

@app.route('/alerts')
def alerts():
    logger.info("Serving alerts page...")
    try:
        with open('alerts.json', 'r') as f:
            alerts = json.load(f)
        return render_template('alerts.html', alerts=alerts)
    except Exception as e:
        logger.error(f"Error serving alerts page: {str(e)}")
        return "Error loading alerts", 500

@app.route('/heatmap')
def heatmap():
    logger.info("Serving heatmap page...")
    return send_file('templates/crime_heatmap.html')

# Main execution
if __name__ == "__main__":
    crime_file = 'telangana_crime_dataset_100k.csv'
    video_path = 'test_trimmed_video.mp4'
    processed_file = os.path.join('outputs', 'processed_crime_data.csv')
    model_output = os.path.join('outputs', 'lstm_model.h5')
    predictions_output = os.path.join('outputs', 'predictions.csv')
    heatmap_output = os.path.join('templates', 'crime_heatmap.html')

    model, history, predictions, crime_map, summaries = crimesense_with_cctv(
        crime_file, video_path, processed_file, model_output, predictions_output, heatmap_output
    )
    if model:
        logger.info("Project completed successfully! Check 'outputs' and 'static' folders for results.")
        app.run(debug=False)