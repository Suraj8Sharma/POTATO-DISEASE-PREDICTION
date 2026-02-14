import streamlit as st
from PIL import Image
import tensorflow as tf 
import numpy as np

#loading the model
@st.cache_resource #it ensures the model is loaded exactly once
def load_potato_model():
    
    model = tf.keras.models.load_model('frontend/model_v1.h5')
    return model
model=load_potato_model()

#the streamlit code 
st.set_page_config(page_title="Potato Disease Classifier", layout="centered")

st.title("🥔 Potato Disease Prediction")
st.write("Identify diseases by uploading an image or taking a live photo.")


if 'input_method' not in st.session_state:
    st.session_state.input_method = None

col1, col2 = st.columns(2)

with col1:
    if st.button("📁 Upload Image", use_container_width=True):
        st.session_state.input_method = "upload"

with col2:
    if st.button("📸 Take Photo", use_container_width=True):
        st.session_state.input_method = "camera"


image = None


if st.session_state.input_method == "upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)


elif st.session_state.input_method == "camera":
    enable = st.checkbox("Enable Camera") 
    
    picture = st.camera_input("Take a picture of the potato leaf", disabled=not enable)
    if picture:
        image = Image.open(picture)


if image is not None:
    st.image(image, caption="Selected Image", use_container_width=True)
    
    if st.button("✨ Predict Disease"):
        with st.spinner("Analyzing...."):
            img_array = np.array(image)

            
            # This turns (height, width, 3) into (1, height, width, 3) because the model is trained on batches
            img_batch = np.expand_dims(img_array, axis=0)
            predicted=model.predict(img_batch)
            pred=np.argmax(predicted)
            class_names = ['Early Blight', 'Late Blight', 'Healthy']
            result_index = pred
            confidence=predicted[0][result_index]
            
            st.success(f"Result: {class_names[result_index]}")
            st.write(f"Confidence: {confidence * 100:.2f}%")
elif st.session_state.input_method is None:
    st.info("Please select an input method above to get started.")