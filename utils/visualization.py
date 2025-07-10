import matplotlib.pyplot as plt
import streamlit as st
class_map = {
    0: "Pepper__bell___Bacterial_spot",
    1: "Pepper__bell___healthy",
    2: "Potato___Early_blight",
    3: "Potato___Late_blight",
    4: "Potato___healthy",
    5: "Tomato_Bacterial_spot",
    6: "Tomato_Early_blight",
    7: "Tomato_Late_blight",
    8: "Tomato_Leaf_Mold",
    9: "Tomato_Septoria_leaf_spot",
    10: "Tomato_Spider_mites_Two_spotted_spider_mite",
    11: "Tomato__Target_Spot",
    12: "Tomato__Tomato_YellowLeaf__Curl_Virus",
    13: "Tomato__Tomato_mosaic_virus",
    14: "Tomato_healthy"
}

def get_class_label(index):
    return class_map.get(index, "Unknown")

def visualize_segmentation(image, heatmap, predicted_class, true_class=None):
    plt.figure(figsize=(12, 4))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Predicted Heatmap")
    plt.axis("off")

    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    title = f"Predicted: {predicted_class}"
    if true_class is not None:
        title += f"\nActual: {true_class}"
    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    st.pyplot(plt.gcf())  # Get current figure and render in Streamlit
    plt.close()  # Optional cleanup

