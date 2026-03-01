import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import subprocess
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Soil Care Agent",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #8B4513;
        margin-top: 1rem;
    }
    .stButton>button {
        background-color: #2E8B57;
        color: white;
        border-radius: 10px;
    }
    .metric-card {
        background-color: #F0F8FF;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LOAD MODELS & DATA
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Loads pre-trained models from disk."""
    try:
        crop_model = joblib.load('crop_model.pkl')
        cnn_model = load_model('soil_cnn.h5')
        forecast_models = joblib.load('forecast_models.pkl')
        return crop_model, cnn_model, forecast_models
    except Exception as e:
        print(f"Failed to load models: {e}")
        return None, None, None

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS (INFERENCE ONLY)
# -----------------------------------------------------------------------------
def predict_crop(model, n, p, k, temp, hum, ph, rain):
    return model.predict([[n, p, k, temp, hum, ph, rain]])[0]

def fertility_check(n, p, k, ph):
    """Advanced rule-based fertility checker."""
    report = {
        "status": "",
        "n_status": "", "p_status": "", "k_status": "", "ph_status": "",
        "action_plan": []
    }
    
    # Nitrogen Analysis
    if n < 50: 
        report["n_status"] = "Low 🔴"
        report["action_plan"].append("Nitrogen is low. Consider adding nitrogen-rich fertilizers like Urea, Ammonium Nitrate, or organic compost like manure.")
    elif n > 120: 
        report["n_status"] = "High 🟢"
        report["action_plan"].append("Nitrogen is high. Avoid adding nitrogenous fertilizers to prevent excessive vegetative growth.")
    else: 
        report["n_status"] = "Optimal 🟡"

    # Phosphorus Analysis    
    if p < 20: 
        report["p_status"] = "Low 🔴"
        report["action_plan"].append("Phosphorus is low. Use Superphosphate or Bone Meal to encourage root development.")
    elif p > 80: 
        report["p_status"] = "High 🟢"
        report["action_plan"].append("Phosphorus levels are sufficient. No need for phosphate fertilizers.")
    else: 
        report["p_status"] = "Optimal 🟡"
    
    # Potassium Analysis
    if k < 20: 
        report["k_status"] = "Low 🔴"
        report["action_plan"].append("Potassium is low. Apply Muriate of Potash (MOP) or wood ash to improve disease resistance.")
    elif k > 80: 
        report["k_status"] = "High 🟢"
        report["action_plan"].append("Potassium is adequate. Excess potassium can inhibit Magnesium absorption.")
    else: 
        report["k_status"] = "Optimal 🟡"

    # pH Analysis
    if ph < 5.5:
        report["ph_status"] = "Acidic 🔴"
        report["action_plan"].append("Soil is Acidic. Apply Lime (Calcium Carbonate) to neutralize acidity.")
    elif ph > 7.5:
        report["ph_status"] = "Alkaline 🔴"
        report["action_plan"].append("Soil is Alkaline. Apply Gypsum or Sulfur to lower pH.")
    else:
        report["ph_status"] = "Neutral 🟢"

    # Overall Score (Simplified logic)
    score = 0
    if "Optimal" in report["n_status"]: score += 25
    if "Optimal" in report["p_status"]: score += 25
    if "Optimal" in report["k_status"]: score += 25
    if "Neutral" in report["ph_status"]: score += 25
    
    report["score"] = score
    if score >= 75: report["status"] = "Excellent 🌱"
    elif score >= 50: report["status"] = "Good 🌿"
    else: report["status"] = "Needs Attention ⚠️"
    
    return report

def forecast_nutrients(models_dict):
    """Forecasting N-P-K levels using saved ID models."""
    years = np.array([2020, 2021, 2022, 2023, 2024]).reshape(-1, 1)
    
    reg_n = models_dict['n_model']
    reg_p = models_dict['p_model']
    reg_k = models_dict['k_model']
    
    future_year = np.array([[2025], [2026], [2027]])
    
    # We reconstruct 'history' from the model for visualization or load real data
    # Here we just predict for simplicity to show the lines
    pred_history_n = reg_n.predict(years)
    pred_history_p = reg_p.predict(years)
    pred_history_k = reg_k.predict(years)
    
    pred_n = reg_n.predict(future_year)
    pred_p = reg_p.predict(future_year)
    pred_k = reg_k.predict(future_year)
    
    return years, pred_history_n, pred_history_p, pred_history_k, future_year, pred_n, pred_p, pred_k

# -----------------------------------------------------------------------------
# GEMINI CHATBOT
# -----------------------------------------------------------------------------
def get_gemini_response(question, api_key, image=None):
    if not api_key:
        return "⚠️ Please provide your Google API Key in the sidebar."
    
    try:
        genai.configure(api_key=api_key)
        # using gemini-1.5-flash as it is multimodal
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        if image:
            inputs = [question, image]
        else:
            inputs = [question]
            
        response = model.generate_content(inputs)
        return response.text
    except Exception as e:
        return f"Error connecting to AI: {str(e)}"

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    crop_model, cnn_model, forecast_models = load_models()
    
    # Check if models are loaded
    if crop_model is None or cnn_model is None or forecast_models is None:
        st.warning("⚠️ Models not found! Generating data and training models now... this may take a moment.")
        try:
            with st.spinner("Running generate_data.py..."):
                subprocess.run([sys.executable, "generate_data.py"], check=True)
            with st.spinner("Running train_models.py..."):
                subprocess.run([sys.executable, "train_models.py"], check=True)
            st.success("✅ Models generated successfully! Reloading...")
            st.rerun()
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Failed to run training scripts. Error: {e}")
            st.stop()
    
    # 2. Sidebar Utils
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)
    st.sidebar.title("Soil Care Utils")
    
    # Get API key from sidebar directly
    api_key = st.sidebar.text_input("Enter your API Key", type="password")
    
    page = st.sidebar.radio("Navigate", [
        "Home", 
        "Crop Recommendation", 
        "Soil Fertility Check", 
        "Nutrient Forecasting", 
        "Soil Image Analysis"
        # "AI Chat Assistant"
    ])
    
    # 3. Pages
    if page == "Home":
        st.markdown('<div class="main-header">🌱 Soil Care Agent</div>', unsafe_allow_html=True)
        # st.write("Welcome to your personal AI Farming Assistant.")
        # st.image("", caption="Smart Farming for Better Future")
        st.sidebar.info("Use the sidebar to access advanced features like Crop Prediction, Soil Analysis, and our AI Doctor.")
        st.header("🤖 AI Farmer Assistant")
        st.write("Ask anything about crops, fertilizers, or soil care! You can also upload an image of your plant or soil.")
        
        col_chat, col_upload = st.columns([2, 1])
        
        with col_chat:
            user_query = st.text_input("Your Question:", "Which crop is best for summers?", key="chat_input")
        
        with col_upload:
            uploaded_image = st.file_uploader("Upload Image (Optional)", type=["jpg", "png", "jpeg"], key="chat_upload")
            
        image = None
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", width=200)

        if st.button("Ask AI"):
            if not api_key:
                st.warning("Please enter your Google API Key in the sidebar to use this feature.")
            else:
                with st.spinner("Thinking..."):
                    if not user_query and not image:
                        st.warning("Please enter a question or upload an image.")
                    else:
                        # Construct prompt
                        prompt = f"You are a helpful agricultural expert assisting a farmer. Answer this simple and clearly: {user_query} Always give response easy to understand in table format if applicable."
                        
                        answer = get_gemini_response(prompt, api_key, image)
                        st.markdown(f"**AI Answer:**\n\n{answer}")


    elif page == "Crop Recommendation":
        st.header("🌾 Crop & Fertilizer Recommendation")
        
        col1, col2 = st.columns(2)
        with col1:
            n = st.number_input("Nitrogen (N)", 0, 150, 50)
            p = st.number_input("Phosphorus (P)", 0, 150, 50)
            k = st.number_input("Potassium (K)", 0, 200, 50)
            temp = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
        with col2:
            hum = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
            ph = st.number_input("pH Level", 0.0, 14.0, 7.0)
            rain = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)
        
        if st.button("Recommend Crop"):
            prediction = predict_crop(crop_model, n, p, k, temp, hum, ph, rain)
            st.success(f"Recommended Crop: **{prediction}**")
            
            # Simple fertilizer logic based on input vs 'ideal' (dummy)
            st.subheader("💊 Fertilizer Suggestion")
            if n < 50:
                st.write("• **Urea**: Apply 50-100 kg/ha to boost Nitrogen.")
            if p < 30:
                st.write("• **DAP**: Apply DAP for phosphorus deficiency.")
            if k < 30:
                st.write("• **MOP**: Potash fertilizer recommended.")
            if n >= 50 and p >= 30 and k >= 30:
                st.write("• Soil nutrient levels seem adequate. Maintain organic compost.")

    elif page == "Soil Fertility Check":
        st.header("🧪 Advanced Soil Fertility Analysis")
        st.write("Analyze soil health based on NPK values and pH level.")
        
        col1, col2 = st.columns(2)
        with col1:
            n = st.slider("Nitrogen (N)", 0, 200, 50, help="Essential for leaf growth")
            p = st.slider("Phosphorus (P)", 0, 200, 40, help="Crucial for root and flower development")
        with col2:
            k = st.slider("Potassium (K)", 0, 200, 40, help="Vital for overall plant health and immunity")
            ph = st.slider("Soil pH Level", 0.0, 14.0, 6.5, help="Measure of acidity/alkalinity")
        
        if st.button("Check Fertility Details"):
            report = fertility_check(n, p, k, ph)
            
            # Display Overall Status
            st.divider()
            st.subheader(f"Overall Health: {report['status']}")
            st.progress(report['score'])
            
            # Detailed Columns
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Nitrogen", f"{n} mg/kg", report['n_status'])
            c2.metric("Phosphorus", f"{p} mg/kg", report['p_status'])
            c3.metric("Potassium", f"{k} mg/kg", report['k_status'])
            c4.metric("pH Level", f"{ph}", report['ph_status'])
            
            # Action Plan
            st.divider()
            st.subheader("📋 Recommended Action Plan")
            if report["action_plan"]:
                for action in report["action_plan"]:
                    st.info(f"• {action}")
            else:
                st.success("Your soil is in perfect condition! Maintain current practices.")
                
            # AI Top 5 Crops
            st.divider()
            st.subheader("🤖 Assistant Recommended Top 5 Crops")
            with st.spinner("Asking AI for best crops based on your soil params..."):
                prompt = f"""
                My soil profile is:
                - Nitrogen (N): {n}
                - Phosphorus (P): {p}
                - Potassium (K): {k}
                - pH Level: {ph}
                
                Based on this specific profile, suggest the TOP 5 crops that would thrive in this soil. 
                Explain briefly why for each. Format as a clean table format, ,be precize avoid writing in high detail.
                """
                ai_recommendation = get_gemini_response(prompt, api_key)
                st.markdown(ai_recommendation)

    elif page == "Nutrient Forecasting":
        st.header("📈 Nutrient Levels Forecast")
        st.write("Predicting future trend based on training data model.")
        
        years, n_h, p_h, k_h, f_years, n_f, p_f, k_f = forecast_nutrients(forecast_models)
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(years, n_h, 'o-', label='Nitrogen (Hist)')
        ax.plot(f_years, n_f, 'x--', label='Nitrogen (Pred)')
        ax.plot(years, p_h, 'o-', label='Phosphorus (Hist)')
        ax.plot(f_years, p_f, 'x--', label='Phosphorus (Pred)')
        ax.set_title("Soil Nutrient Trends (2020-2027)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Level (kg/ha)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    elif page == "Soil Image Analysis":
        st.header("📸 Visual Soil Analysis (CNN)")
        st.write("Upload an image of your soil. The CNN model will classify moisture level.")
        
        uploaded_file = st.file_uploader("Choose a soil image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Soil Image', width=300)
            
            # Preprocess for CNN (Resize to 64x64)
            img_array = image.resize((64, 64))
            img_array = np.array(img_array) / 255.0
            
            if img_array.shape == (64, 64, 3):
                img_array = np.expand_dims(img_array, axis=0) # Batch dim
                
                prediction = cnn_model.predict(img_array)
                class_names = ['Dry Soil', 'Normal Soil', 'Wet/Muddy Soil']
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction) * 100
                
                st.subheader("Analysis Result")
                st.metric("Soil Condition", predicted_class, f"{confidence:.1f}% Confidence")
                
                if predicted_class == 'Dry Soil':
                    st.warning("💧 Action: Irrigation needed immediately.")
                elif predicted_class == 'Wet/Muddy Soil':
                    st.info("☀️ Action: Allow drainage. Avoid over-watering.")
                else:
                    st.success("✅ Action: Soil moisture is optimal.")
            else:
                st.error("Error: Image must be RGB.")

    elif page == "AI Chat Assistant":
        pass
if __name__ == "__main__":
    main()
