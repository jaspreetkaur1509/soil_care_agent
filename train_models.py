import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

def train_crop_model():
    print("Training Crop Recommendation Model...")
    df = pd.read_csv('crop_data.csv')
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, 'crop_model.pkl')
    print("✓ Crop model saved to crop_model.pkl")

def train_soil_cnn():
    print("Training Soil Analysis CNN (from soil_data.csv)...")
    # Load flattened data
    df = pd.read_csv('soil_data.csv')
    X_flat = df.drop('label', axis=1).values
    y = df['label'].values
    
    # Reshape back to images: (N, 64, 64, 3)
    # The csv columns were flattened R, G, B pixels
    img_size = 64
    X = X_flat.reshape(-1, img_size, img_size, 3)
    X = X / 255.0  # Normalize
    
    # Train simple CNN
    model = models.Sequential([
        layers.Input(shape=(img_size, img_size, 3)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax') # 3 classes: Dry, Normal, Wet
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    
    model.save('soil_cnn.h5')
    print("✓ CNN model saved to soil_cnn.h5")

def train_forecast_models():
    print("Training Nutrient Forecasting Models...")
    # Synthetic time-series for training
    years = np.array([2020, 2021, 2022, 2023, 2024]).reshape(-1, 1)
    n_trend = np.array([120, 115, 110, 108, 105])
    p_trend = np.array([50, 48, 45, 42, 40])
    k_trend = np.array([45, 44, 43, 41, 40])
    
    reg_n = LinearRegression().fit(years, n_trend)
    reg_p = LinearRegression().fit(years, p_trend)
    reg_k = LinearRegression().fit(years, k_trend)
    
    # Save as dictionary
    forecast_models = {
        'n_model': reg_n,
        'p_model': reg_p,
        'k_model': reg_k
    }
    joblib.dump(forecast_models, 'forecast_models.pkl')
    print("✓ Forecast models saved to forecast_models.pkl")

if __name__ == "__main__":
    if not os.path.exists('crop_data.csv'):
        print("❌ Error: crop_data.csv not found. Run generate_data.py first.")
    else:
        train_crop_model()
        train_soil_cnn()
        train_forecast_models()
        print("\nAll models trained and saved.")
