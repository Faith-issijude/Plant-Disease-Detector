# Plant-Disease-Detector

# 🌿 Plant Doctor - AI Leaf Disease Diagnosis App

Plant Doctor is a powerful Streamlit-based web app that allows users (especially farmers and researchers) to detect and visualize crop leaf diseases using a hybrid CNN–Transformer deep learning model. It supports classification and diagnosis of tomato, potato, and pepper diseases, along with treatment suggestions powered by Cohere's language model.

---

## 🚀 Features

- CNN + MobileViT-based Transformer for high-accuracy image classification
- Grad-CAM visualizations to show disease focus areas
- Upload any crop leaf image for diagnosis
- AI Assistant with Cohere API for treatment advice
- Disease browsing by plant type (tomato, potato, pepper)
- Real-time prediction confidence display

---

## 🖼 Supported Diseases

Supports 15 common crop conditions including:

- Tomato: Early Blight, Leaf Mold, Yellow Leaf Curl Virus, Mosaic Virus, etc.
- Potato: Early Blight, Late Blight, Healthy
- Pepper (Bell): Bacterial Spot, Healthy

---

## 🛠 Tech Stack

- **Frontend:** Streamlit
- **Backend:** TensorFlow (CNN + MobileViT hybrid)
- **Visualization:** Grad-CAM, Matplotlib
- **NLP Assistant:** Cohere API
- **Data Preprocessing:** Keras ImageDataGenerator, OpenCV

---


## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/plant-doctor-ai.git
   cd plant-doctor-ai

2. **Create and activate a virtual environment**
  ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows

3. **Install requirements**
  ```bash
   pip install -r requirements.txt

4. **Run the app**
  ```bash
   streamlit run main.py

## Example
Upload a sample tomato leaf image and click Diagnose Leaf Disease to see:

Prediction label: Tomato Late Blight

Confidence: 95.3%

Heatmap showing infection region

Suggested treatment from AI assistant


## 📜 License
This project is licensed under the MIT License. See LICENSE for more information.

## 🔗 Acknowledgements
PlantVillage Dataset
Cohere Language API
TensorFlow, Streamlit, OpenCV, and the ML community 🌍
