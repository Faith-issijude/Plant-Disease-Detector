import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from utils.gradcam import get_gradcam_heatmap
from utils.visualization import visualize_segmentation, get_class_label

st.set_page_config(page_title="🌿 Plant Doctor", layout="wide")
st.sidebar.header("🧭 Navigation")
section = st.sidebar.radio("Go to", ["🌿 Upload Your Plant", "🌱 Plants & Diseases", "🤖 Crop Assistant"])

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models\recent_model.keras", compile=False)

def preprocess_image(image):
    img = np.array(image.resize((224, 224)))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

try:    
    model = load_model()
except Exception as e:
    model = None
    st.error(f"❌ Failed to load model: {e}")


if section == "🌿 Upload Your Plant":
    st.markdown("### 🌾 **Upload your crop image below**")
    st.markdown("💬 _Chat-style upload (like ChatGPT)_")
    st.markdown("**📤 Upload your leaf image here:**")
    uploaded_file = st.file_uploader("Upload", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Leaf", use_column_width=True)

        input_image = preprocess_image(image)
        prediction = model.predict(input_image)
        pred_index = np.argmax(prediction)
        pred_label = get_class_label(pred_index)

        st.success(f"🩺 Predicted Disease: **{pred_label}**")

        try:
            heatmap = get_gradcam_heatmap(input_image, model, last_conv_layer_name='last_conv', pred_index=pred_index)
            heatmap_resized = cv2.resize(heatmap, (224, 224))
            visualize_segmentation(np.array(image.resize((224, 224))), heatmap_resized, pred_label)
        except Exception as e:
            st.warning(f"⚠️ Could not generate Grad-CAM: {e}")

    else:
        user_text = st.chat_input("Ask about your plant (e.g., 'Why are my leaves yellow?')")
        if user_text:
            st.chat_message("user").write(user_text)
            st.chat_message("ai").write("🧠 Suggestion: This is a mock response. Consider using neem-based pesticide or adjusting NPK levels.")

elif section == "🌱 Plants & Diseases":
    st.markdown("### 🌿 Plants in Dataset")
    plant_classes = ["Tomato", "Potato", "Pepper Bell"]
    st.markdown("- " + "\n- ".join(plant_classes))

    st.markdown("### 🦠 Diseases Included")
    disease_classes = [
        "Early Blight", "Late Blight", "Leaf Mold", "Bacterial Spot",
        "Yellow Leaf Curl Virus", "Mosaic Virus", "Septoria Leaf Spot"
    ]
    st.markdown("- " + "\n- ".join(disease_classes))

elif section == "🤖 Crop Assistant":
    st.markdown("### 🤖 Ask the AI Assistant")
    question = st.text_input("What issue are you facing with your crop?")
    if question:
        st.success("This is a mock answer. You can link this to a real AI API.")
        st.write("🧠 Suggestion: Use neem-based pesticides for organic fungal protection.")
