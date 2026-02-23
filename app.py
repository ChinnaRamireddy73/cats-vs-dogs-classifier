import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Paws & Whiskers Classifier",
    page_icon="🐾",
    layout="centered"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
        border: none;
    }
    .result-text {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("🐾 App Info")
    
    st.markdown("### 📖 About")
    st.write("This application uses Deep Learning to distinguish between our favorite furry friends: Cats and Dogs! 🐱🐶")
    
    st.markdown("### 🛠️ Instructions")
    st.write("1. Upload an image of a cat or a dog.")
    st.write("2. Wait for the image to display.")
    st.write("3. Click the **'Predict'** button.")
    st.write("4. See the magic happen!")

    st.markdown("### 🧠 Model Info")
    st.info("**Architecture:** MobileNetV2\n\n**Technique:** Transfer Learning (Fine-tuned)\n\n**Input Size:** 224x224 px")

    st.markdown("### 👤 Author")
    st.write("Developed with ❤️ by [Your Name/Antigravity]")
    st.divider()
    st.write("© 2026 Paws & Whiskers AI")

# --- Main App ---
st.title("🐱 Dogs vs Cats Classifier 🐶")
st.markdown("---")
st.write("Welcome to the professional image classifier! Just upload a photo, and our AI will tell you if it's a playful pup or a fancy feline.")

# --- Model Loading ---
@st.cache_resource
def load_model():
    model_path = "cats_vs_dogs_model.keras"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error(f"🚨 Model file '{model_path}' not found! Please ensure it's in the project directory.")
        return None

model = load_model()

# --- Image Preprocessing ---
def preprocess_image(image):
    # Resize to 224x224
    img = image.resize((224, 224))
    # Convert to array
    img_array = np.array(img)
    
    # Ensure 3 channels (RGB)
    if img_array.shape[-1] != 3:
        img = img.convert('RGB')
        img_array = np.array(img)
        
    # Normalize by dividing by 255
    img_array = img_array.astype('float32') / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- UI Components ---
uploaded_file = st.file_uploader("Choose a Cat or Dog image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        predict_btn = st.button("🔍 Predict")
        
        if predict_btn:
            if model is not None:
                with st.spinner('Analyzing the image...'):
                    # Preprocess
                    processed_img = preprocess_image(image)
                    
                    # Predict
                    prediction = model.predict(processed_img)
                    
                    # Assuming a binary classification or 2 classes (Dog = 1, Cat = 0 often, depends on training)
                    # Let's check prediction shape to be safe
                    if prediction.shape[-1] > 1:
                        # Multi-class format
                        score = tf.nn.softmax(prediction[0])
                        class_idx = np.argmax(score)
                        confidence = 100 * np.max(score)
                        # Map indices (standard for cats vs dogs: Cat=0, Dog=1)
                        result_label = "Dog" if class_idx == 1 else "Cat"
                    else:
                        # Binary format (Sigmoid)
                        prob = prediction[0][0]
                        if prob > 0.5:
                            result_label = "Dog"
                            confidence = prob * 100
                        else:
                            result_label = "Cat"
                            confidence = (1 - prob) * 100

                # --- Results Display ---
                if result_label == "Dog":
                    st.success(f"### It's a **DOG**! 🐶")
                    st.write(f"Confidence: **{confidence:.2f}%**")
                    st.balloons()
                    st.info("🦴 *Who's a good boy? This paws-itive result just made our day! Ready for a walk?*")
                else:
                    st.success(f"### It's a **CAT**! 🐱")
                    st.write(f"Confidence: **{confidence:.2f}%**")
                    st.snow()
                    st.info("🐾 *Meow-tastic! This kitty looks purr-fect. Time for a nap or world domination?*")
            else:
                st.warning("Model is not loaded. Cannot perform prediction.")
                
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
else:
    st.info("Please upload an image file to start the classification.")
