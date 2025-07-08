import numpy as np
import cv2
from tensorflow.keras.models import load_model
from utils.custom_layers import PatchExtract


def load_trained_model(path):
    return load_model(path, compile=False, custom_objects={"PatchExtract": PatchExtract})


def preprocess_image(image, size=(224, 224)):
    image = cv2.resize(image, size)
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)


def predict(model, image_np):
    img_batch = preprocess_image(image_np)
    preds = model.predict(img_batch)
    class_index = np.argmax(preds[0]) if isinstance(preds, list) else np.argmax(preds)
    confidence = preds[0][class_index] if isinstance(preds, list) else preds[class_index]
    return class_index, confidence
