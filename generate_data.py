import pandas as pd
import numpy as np
import os

def generate_crop_data():
    """Generates synthetic crop recommendation dataset."""
    print("Generating crop_data.csv...")
    n_samples = 2000
    np.random.seed(42)  # For reproducibility
    
    data = {
        'N': np.random.randint(0, 140, n_samples),
        'P': np.random.randint(5, 145, n_samples),
        'K': np.random.randint(5, 205, n_samples),
        'temperature': np.random.uniform(8.8, 38, n_samples),
        'humidity': np.random.uniform(14, 100, n_samples),
        'ph': np.random.uniform(3.5, 9.9, n_samples),
        'rainfall': np.random.uniform(20, 298, n_samples)
    }
    
    labels = []
    for i in range(n_samples):
        # Slightly more complex rules for diversity
        if data['N'][i] > 120: labels.append('Cotton')
        elif data['P'][i] > 120: labels.append('Grapes')
        elif data['K'][i] > 150: labels.append('Apple')
        elif data['rainfall'][i] > 200: labels.append('Rice')
        elif data['temperature'][i] < 15: labels.append('Wheat')
        elif data['temperature'][i] > 35: labels.append('Moth Beans')
        elif data['ph'][i] < 5: labels.append('Coffee')
        elif data['humidity'][i] < 20: labels.append('Chickpea')
        else: labels.append('Maize')
    
    df = pd.DataFrame(data)
    df['label'] = labels
    df.to_csv('crop_data.csv', index=False)
    print("✓ crop_data.csv created.")

def generate_fertilizer_data():
    """Generates synthetic fertilizer dataset."""
    print("Generating fertilizer_data.csv...")
    fert_data = {
        'Crop': ['Rice', 'Wheat', 'Maize', 'Cotton', 'Apple', 'Grapes', 'Coffee', 'Moth Beans', 'Chickpea'],
        'N_needed': [80, 60, 100, 120, 20, 20, 100, 30, 40],
        'P_needed': [40, 30, 60, 60, 125, 125, 20, 40, 50],
        'K_needed': [40, 30, 40, 40, 200, 200, 30, 30, 40],
        'Fertilizer': ['Urea', 'DAP', 'NPK 14-35-14', 'Urea', 'MOP', 'MOP', 'Urea', 'DAP', 'DAP']
    }
    pd.DataFrame(fert_data).to_csv('fertilizer_data.csv', index=False)
    print("✓ fertilizer_data.csv created.")

def generate_soil_image_data():
    """Generates synthetic soil image data (flattened pixels) for CNN training."""
    print("Generating soil_data.csv (Synthetic Image Data)...")
    num_samples = 200
    img_size = 64
    
    # We will save flattened images: 64*64*3 = 12288 columns + 1 label column
    # This simulates a dataset where images are pre-processed into a CSV
    
    X = []
    y = []
    
    for _ in range(num_samples):
        # Decide class
        label = np.random.randint(0, 3) # 0: Dry, 1: Normal, 2: Wet
        
        if label == 0: # Dry (Light Brown/Yellowish)
            img = np.random.randint(160, 255, (img_size, img_size, 3), dtype=np.uint8) 
            img[:, :, 2] = np.random.randint(80, 140, (img_size, img_size)) # Low Blue
        elif label == 1: # Normal (Brown)
            img = np.random.randint(100, 180, (img_size, img_size, 3), dtype=np.uint8)
            img[:, :, 2] = np.random.randint(40, 90, (img_size, img_size)) 
        else: # Wet (Dark/Black)
            img = np.random.randint(30, 90, (img_size, img_size, 3), dtype=np.uint8)
            img[:, :, 2] = np.random.randint(0, 40, (img_size, img_size))
            
        X.append(img.flatten())
        y.append(label)
    
    # Create DataFrame (might be large, but acceptable for 200 samples)
    cols = [f'pixel_{i}' for i in range(img_size*img_size*3)]
    df = pd.DataFrame(X, columns=cols)
    df['label'] = y
    
    # Save as CSV (compressed to save space if needed, but standard csv for now)
    df.to_csv('soil_data.csv', index=False)
    print("✓ soil_data.csv created.")

if __name__ == "__main__":
    generate_crop_data()
    generate_fertilizer_data()
    generate_soil_image_data()
    print("\nAll datasets generated successfully.")
