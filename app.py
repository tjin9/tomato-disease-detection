import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load our trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('tomato_disease_model.keras')
    return model

model = load_model()

# List of the 10 tomato classes (in exact order used during training)
class_names = [
    'Tomato___Target_Spot',
    'Tomato___Bacterial_spot',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___healthy',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___Leaf_Mold',
    'Tomato___Spider_mites Two-spotted_spider_mite'
]

# UI Design
st.title("🌿 Tomato Plant Disease Detection")
st.write("Upload an image of a tomato leaf to check if it's healthy or affected by a disease.")

uploaded_file = st.file_uploader("Choose a tomato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='Uploaded Tomato Leaf', use_container_width=True)
    
    st.write("Classifying...")
    
    # Preprocess the image for the model
    img = image_display.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Convert to batch dimension (1, 224, 224, 3)
    img_array = img_array / 255.0 # Rescale pixels
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    predicted_class_name = class_names[predicted_class_index]
    
    # Show results
    st.success(f"**Prediction:** {predicted_class_name}")
    st.info(f"**Confidence:** {confidence:.2f}%")
