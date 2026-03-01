# 🌱 Soil Care Agent - Operational Manual

This manual provides instructions on setting up and using the Soil Care Agent application.

## 1. Project Overview
The **Soil Care Agent** is an AI-powered Streamlit application designed to assist farmers with:
- **Crop & Fertilizer Recommendations**: Uses Machine Learning to suggest optimal crops.
- **Soil Fertility Check**: Analyzes N-P-K levels.
- **Nutrient Forecasting**: Predicts future soil health trends.
- **Visual Soil Analysis**: Uses a custom CNN model to detect soil moisture from images.
- **AI Farmer Assistant**: A chatbot powered by Google's Gemini AI.

## 2. Prerequisites & Installation

Ensure you have Python installed. Install the required libraries:

```bash
pip install -r requirements.txt
```

### API Key Setup
For the AI Chatbot to work, you need a Google Gemini API Key.
The app reads this from a `.env` file.
1.  Open `.env` file.
2.  Ensure your key is set: `GOOGLE_API_KEY=AIzaSy...`

## 3. How to Run

### Step 1: Generate Data
Creates the synthetic CSV datasets (`crop_data.csv`, `fertilizer_data.csv`, `soil_data.csv`) if they don't exist.
```bash
python generate_data.py
```

### Step 2: Train Models
Trains the Machine Learning models using the generated data and saves them (`.pkl`, `.h5`).
```bash
python train_models.py
```

### Step 3: Run Application
Launches the Streamlit app.
```bash
streamlit run app.py
```

The application will launch in your default web browser (usually at `http://localhost:8501`).

## 4. Features Guide

### 🌾 Crop Recommendation
1. Enter the soil params (N, P, K, pH) and weather info.
2. Click **Recommend Crop**.
3. The app will predict the best crop and suggest fertilizers.

### 🧪 Soil Fertility Check
1. Use sliders to set current Nitrogen, Phosphorus, and Potassium levels.
2. Click **Check Fertility**.
3. View whether your soil is Low, Medium, or High fertility with specific flags.

### 📈 Nutrient Forecasting
1. Navigate to this page to see a graph.
2. The blue lines show historical trends (synthetic data), and orange x-marks show the prediction for the next 3 years using Linear Regression.
The "Nutrient Forecasting" page (in the sidebar) contains a predictive analysis tool for soil health.

Here is what it specifically does:

Forecasts Future Levels:
It uses Linear Regression (a Machine Learning forecasting technique) to analyze historical soil data trends.
It predicts the levels of Nitrogen (N), Phosphorus (P), and Potassium (K) for the next 3 years (2025, 2026, 2027).
Visual Graph:
It displays a line chart showing the trend over time.
Solid Lines (O): Represent historical data (2020-2024).
Dashed Lines (X): Represent the predicted future values.
Purpose: This helps farmers anticipate if their soil quality is degrading over time so they can plan long-term fertilizer strategies before the soil becomes too depleted.


### 📸 Visual Soil Analysis (CNN)
1. Upload a clear image of soil.
2. The system processes it using a lightweight Convolutional Neural Network (CNN).
3. Result: **Dry**, **Normal**, or **Wet/Muddy**.

### 🤖 AI Chat Assistant
1. Type a question like *"How much urea for wheat?"*.
2. You can also upload an image of a plant or soil for the AI to analyze.
3. Click **Ask AI**.
4. The Gemini AI provides a simple, farmer-friendly answer.

## 5. File Structure & Detailed Explanation

Here is a comprehensive breakdown of every file in the project directory, explaining its purpose and content.

### 📁 Core Application Files

#### `app.py`
This is the **main application script**. You run this file to start the web app.
- **Libraries**: Uses `streamlit` for the UI, `pandas`/`numpy` for data, `tensorflow` for the CNN, `sklearn` for ML, and `google.generativeai` for the Chatbot.
- **Configuration**: Sets up the page title, icon, and sidebar navigation.
- **Model Loading**: Automatically loads the saved brains (`crop_model.pkl`, `soil_cnn.h5`, `forecast_models.pkl`) at startup.
- **Pages**:
    - **Home**: Displays the Welcome screen and the **AI Chat Assistant** interface (User Query + Image Upload -> Gemini AI).
    - **Crop Recommendation**: Inputs N-P-K-Temp-Humidity-pH-Rainfall -> Outputs predicted Crop.
    - **Soil Fertility Check**: Inputs Soil Params -> Outputs Health Score, pH Status, and AI Recommendations.
    - **Nutrient Forecasting**: Displays graphs predicting future soil nutrient levels.
    - **Soil Image Analysis**: Accepts an image upload -> Resizes it -> Uses CNN to predict (Dry / Normal / Wet).

#### `requirements.txt`
This file lists all the **external Python libraries** required to run the project.
- Used by `pip install -r requirements.txt`.
- Includes: `streamlit`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, `google-generativeai`, `Pillow` (for images), `python-dotenv`.

#### `.env`
This is a **configuration file** for storing sensitive secrets.
- **Content**: It holds your `GOOGLE_API_KEY`.
- **Reason**: We don't hardcode API keys in `app.py` for security. The app loads this file using `python-dotenv`.

---

### 📁 Data Generation & Model Training Scripts

#### `generate_data.py`
This runs **Step 1** of the setup. It creates synthetic (fake but realistic) data for demonstration purposes.
- **`generate_crop_data()`**: Creates `crop_data.csv` with 2000 rows of random soil/weather conditions and labels them based on logic (e.g., High Rainfall -> Rice).
- **`generate_fertilizer_data()`**: Creates `fertilizer_data.csv` with a simple mapping of Crops to Fertilizer recommendations.
- **`generate_soil_image_data()`**: Creates `soil_data.csv`. Instead of folders of images, it generates mathematical representations (flattened arrays) of "brown" pixels to simulate soil images for the CNN.

#### `train_models.py`
This runs **Step 2** of the setup. It learns from the data and saves the "brains" (models).
- **`train_crop_model()`**: Reads `crop_data.csv`, trains a **Random Forest Classifier**, and saves it as `crop_model.pkl`.
- **`train_soil_cnn()`**: Reads `soil_data.csv`, extracts the pixel data, reshapes it into 64x64 images, trains a **Convolutional Neural Network (CNN)** using TensorFlow, and saves it as `soil_cnn.h5`.
- **`train_forecast_models()`**: Trains **Linear Regression** models on hardcoded historical trends and saves them as `forecast_models.pkl`.

---

### 📁 Dataset Files (Generated)

#### `crop_data.csv`
- **Purpose**: Training data for the Crop Recommender.
- **Structure**: Columns for N, P, K, temperature, humidity, ph, rainfall, and **label** (the answer, e.g., "Cotton").

#### `fertilizer_data.csv`
- **Purpose**: Lookup table for fertilizer advice.
- **Structure**: Maps Crop names to N-P-K requirements and suggested Fertilizer names.

#### `soil_data.csv`
- **Purpose**: Training data for the Image Analysis CNN.
- **Structure**: A very wide CSV. Each row is one image. Columns `pixel_0` to `pixel_12287` represent the colors of the image. The `label` column indicates if it is Dry (0), Normal (1), or Wet (2).

---

### 📁 Trained Model Files (Artifacts)

#### `crop_model.pkl`
- **What it is**: The "pickled" (saved) Random Forest model.
- **Role**: Does the math to convert your 7 input numbers (N, P, K, etc.) into a Crop Name.

#### `soil_cnn.h5`
- **What it is**: The saved Keras/TensorFlow Deep Learning model.
- **Role**: Contains the "weights" of the neural network that can look at a 64x64 image grid and decide if it looks like Dry, Wet, or Normal soil.

#### `forecast_models.pkl`
- **What it is**: A dictionary containing three small Linear Regression models.
- **Role**: One model predicts Nitrogen trend, one for Phosphorus, and one for Potassium. Used to draw the dotted lines on the forecast graph.

## 6. Troubleshooting

- **Missing Data Files**: If `crop_data.csv` or models are missing, run `python generate_data.py` followed by `python train_models.py`.
- **Image Error**: Ensure uploaded images are standard RGB (JPG/PNG).
- **Chatbot Not Working**: Verify your `GOOGLE_API_KEY` is set correctly in `.env` and you have internet access. If you see a "404 Model Not Found" error, ensure `app.py` is using `gemini-1.5-flash`.
