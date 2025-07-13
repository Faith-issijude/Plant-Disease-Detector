import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from utils.gradcam import get_gradcam_heatmap
from utils.visualization import visualize_segmentation, get_class_label
from utils.custom_layers import PatchExtract
from utils.api_assistant import cohere_chat

# Initialize session state
if "section" not in st.session_state:
    st.session_state.section = "Upload Your Plant"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "redirect_to_chat" not in st.session_state:
    st.session_state.redirect_to_chat = False
if "chat_input_value" not in st.session_state:
    st.session_state.chat_input_value = None
if "chat_question" not in st.session_state:
    st.session_state.chat_question = ""
if "upload_state" not in st.session_state:
    st.session_state.upload_state = {
        "file": None,
        "prediction": None,
        "confidence": None,
        "label": None
    }

st.set_page_config(page_title="Plant Doctor", layout="wide")

# Sidebar navigation
available_sections = ["Upload Your Plant", "Plants & Diseases", "Crop Assistant"]
if st.session_state.redirect_to_chat:
    st.session_state.redirect_to_chat = False
    st.session_state.section = "Crop Assistant"
    if st.session_state.chat_input_value:
        st.session_state.chat_history.append(("user", st.session_state.chat_input_value))
        with st.spinner("Thinking..."):
            ai_reply = cohere_chat(f"Give advice on this plant issue: {st.session_state.chat_input_value}")
        st.session_state.chat_history.append(("ai", ai_reply))
        st.session_state.chat_input_value = None

section = st.sidebar.radio("Go to", available_sections, index=available_sections.index(st.session_state.section))
st.session_state.section = section

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/recent_model.keras", custom_objects={"PatchExtract": PatchExtract}, compile=False)

def preprocess_image(image):
    img = np.array(image.resize((224, 224)))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

try:
    model = load_model()
except Exception as e:
    model = None
    st.error(f"Failed to load model: {e}")

if section == "Upload Your Plant":
    st.markdown("### **Plant Doctor**")
    st.markdown("_Upload your crop image below_")
    st.markdown("**Upload your leaf image here:**")

    uploaded_file = st.file_uploader("Upload", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file:
        st.session_state.upload_state["file"] = uploaded_file

    if st.session_state.upload_state["file"] and model:
        image = Image.open(st.session_state.upload_state["file"]).convert("RGB")
        st.image(image, caption="Uploaded Leaf", width=400)

        if st.session_state.upload_state["prediction"] is None:
            input_image = preprocess_image(image)
            try:
                prediction = model.predict(input_image)
                pred_index = np.argmax(prediction)
                pred_label = get_class_label(pred_index)
                confidence = float(np.max(prediction)) * 100

                st.session_state.upload_state.update({
                    "prediction": prediction,
                    "confidence": confidence,
                    "label": pred_label
                })
            except Exception as e:
                st.error(f"Model prediction failed: {e}")

        if st.session_state.upload_state["label"]:
            if st.button("🔍 Diagnose Leaf Disease", key="diagnosis_button"):
                st.success(f"Predicted Disease: **{st.session_state.upload_state['label']}**")
                st.markdown(f"**Confidence:** {st.session_state.upload_state['confidence']:.2f}%")
                try:
                    input_image = preprocess_image(image)
                    heatmap = get_gradcam_heatmap(input_image, model, last_conv_layer_name='last_conv', pred_index=np.argmax(st.session_state.upload_state["prediction"]))
                    heatmap_resized = cv2.resize(heatmap, (224, 224))
                    visualize_segmentation(np.array(image.resize((224, 224))), heatmap_resized, st.session_state.upload_state['label'])
                except Exception as e:
                    st.warning(f"Could not generate Grad-CAM: {e}")

            if st.button("💊 Suggest Treatment", key="remedy_button"):
                with st.spinner("Contacting AI Assistant for treatment suggestion..."):
                    advice = cohere_chat(f"Suggest a treatment for: {st.session_state.upload_state['label']}")
                    st.success(f"🧪 Suggested Remedy: {advice}")

        if st.button("🔁 Reset Upload Section"):
            st.session_state.upload_state = {"file": None, "prediction": None, "confidence": None, "label": None}
            st.rerun()

    user_text = st.chat_input("Ask about your plant (e.g., 'Why are my leaves yellow?')")
    if user_text:
        st.session_state.chat_input_value = user_text
        st.session_state.redirect_to_chat = True
        st.rerun()

elif section == "Plants & Diseases":
    st.markdown("### Browse by Plant")
    plant_options = {
        "Tomato": [
            "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight",
            "Tomato Leaf Mold", "Tomato Septoria Leaf Spot",
            "Tomato Spider Mites Two Spotted Spider Mite", "Tomato Target Spot",
            "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus", "Tomato Healthy"
        ],
        "Potato": ["Potato Early Blight", "Potato Late Blight", "Potato Healthy"],
        "Pepper Bell": ["Pepper Bell Bacterial Spot", "Pepper Bell Healthy"]
    }
    plant_choice = st.selectbox("Select a plant:", list(plant_options.keys()))

    st.markdown("### Related Diseases:")
    for disease in plant_options[plant_choice]:
        st.markdown(f"- {disease}")

elif section == "Crop Assistant":
    st.markdown("### 🤖 Ask the AI Assistant")

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []

    chat_input = st.chat_input("Ask about your plant (e.g., 'Why are my leaves yellow?')")
    if chat_input:
        st.session_state.chat_history.append(("user", chat_input))
        with st.spinner("Thinking..."):
            ai_reply = cohere_chat(f"Give advice on this plant issue: {chat_input}")
        st.session_state.chat_history.append(("ai", ai_reply))

    for sender, msg in st.session_state.chat_history:
        st.chat_message(sender).write(msg)
