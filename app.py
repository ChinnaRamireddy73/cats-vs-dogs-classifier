import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import time
import pandas as pd
import io

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Paws & Whiskers AI",
    page_icon="🐾",
    layout="wide"
)

# ---------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------
if 'history' not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------
# FIXED COLORS & STYLING
# ---------------------------------------------------
bg = "#0b1220"
card = "#111827"
text = "#f9fafb"
sub = "#9ca3af"
border = "#1f2937"
accent = "#0071e3"

st.markdown(f"""
<style>
    .stApp {{ background-color: {bg}; color: {text}; }}
    .title-text {{ text-align: center; font-size: 48px; font-weight: 800; color: {text}; margin-bottom: 5px; }}
    .subtitle-text {{ text-align: center; color: {sub}; font-size: 20px; margin-bottom: 40px; }}
    
    /* Card Styling */
    .custom-card {{
        background: {card};
        padding: 25px;
        border-radius: 20px;
        border: 1px solid {border};
        box-shadow: 0 15px 35px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }}
    
    /* Result Box */
    .result-container {{
        background: rgba(0, 113, 227, 0.1);
        border: 2px solid {accent};
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 20px;
        justify-content: center;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        font-weight: 600;
        font-size: 18px;
    }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/dog.png", width=100)
    st.markdown("# Paws & Whiskers AI")
    st.write("Professional-grade image classification system powered by MobileNetV2 Transfer Learning.")
    
    st.divider()
    
    st.markdown("### 🗂️ History")
    if not st.session_state.history:
        st.write("No predictions yet.")
    else:
        for item in reversed(st.session_state.history[-5:]):
            st.write(f"• **{item['label']}** ({item['conf']:.1f}%)")
    
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.markdown("### 🛠️ Tech Stack")
    st.caption("TensorFlow 2.x • Streamlit • Pillow")

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown('<div class="title-text">Cats vs Dogs AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">State-of-the-art vision system for our furry companions</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------
@st.cache_resource
def load_model():
    path = "cats_vs_dogs_model.keras"
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None

def preprocess(img):
    # Standard resize and normalization for MobileNetV2
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0), img_resized

# ---------------------------------------------------
# MAIN CONTENT (TABS)
# ---------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🚀 Classifier", "📜 Full History", "🧠 Deep Dive"])

model = load_model()

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload Image")
        uploaded = st.file_uploader("Drop a JPG or PNG here", type=["jpg", "jpeg", "png"])
        
        st.markdown("### 🧪 Quick Test Samples")
        sample_col1, sample_col2 = st.columns(2)
        # Note: In a real app, these would be local assets or stable URLs
        if sample_col1.button("🐕 Load Dog Sample"):
            # Placeholder for logic to load a dog image
            st.info("Direct sample loading requires local asset paths.")
        if sample_col2.button("🐈 Load Cat Sample"):
            st.info("Direct sample loading requires local asset paths.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Current Input", use_container_width=True)
            
            if st.button("Analyze Image", use_container_width=True):
                if model:
                    with st.spinner("Classifying..."):
                        time.sleep(1.0) # UX Delay
                        processed_tensor, _ = preprocess(image)
                        prediction = model.predict(processed_tensor)[0][0]
                        
                        label = "Dog" if prediction > 0.5 else "Cat"
                        conf = (prediction if prediction > 0.5 else 1 - prediction) * 100
                        
                        # Save to history
                        st.session_state.history.append({"label": label, "conf": conf, "time": time.strftime("%H:%M:%S")})
                        
                        # Display Result
                        st.markdown(f"""
                        <div class="result-container">
                            <h2 style='margin:0;'>{label} {'🐶' if label == "Dog" else '🐱'}</h2>
                            <p style='margin:0; font-size:18px;'>Confidence Points: {conf:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.progress(int(conf))
                        
                        if label == "Dog":
                            st.balloons()
                            st.success("Friendly canine detected! 🦴")
                        else:
                            st.snow()
                            st.success("Graceful feline detected! 🐾")
                            
                        # Download Report
                        report_text = f"Classification Report\nTarget: {label}\nConfidence: {conf:.2f}%\nTimestamp: {time.ctime()}"
                        st.download_button(
                            label="📥 Download Report",
                            data=report_text,
                            file_name=f"{label}_report.txt",
                            mime="text/plain"
                        )
                else:
                    st.error("Model not available.")
        else:
            st.info("Upload an image on the left to start.")

with tab2:
    st.markdown("### 📊 Prediction Logs")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.table(df)
        
        # Plotting (basic)
        if len(st.session_state.history) > 1:
            st.line_chart(df['conf'])
    else:
        st.write("No history available yet. Try the classifier!")

with tab3:
    st.markdown("### 🔬 Model Technicals")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        processed_tensor, resized_img = preprocess(image)
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.write("**What the Model Sees:**")
            st.image(resized_img, caption="224x224 input space", width=224)
        
        with col_t2:
            st.write("**Data Structure:**")
            st.code(f"Shape: {processed_tensor.shape}\nRange: [{processed_tensor.min():.2f}, {processed_tensor.max():.2f}]")
    
    st.markdown("""
    #### Architecture Overview
    - **Backbone:** MobileNetV2 (Pre-trained on ImageNet)
    - **Optimization:** Global Average Pooling
    - **Classifier:** Dense (1 unit, Sigmoid)
    - **Resolution:** 224x224 RGB
    """)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.divider()
st.markdown('<div style="text-align: center; color: #9ca3af;">Developed for Paws & Whiskers AI © 2026</div>', unsafe_allow_html=True)