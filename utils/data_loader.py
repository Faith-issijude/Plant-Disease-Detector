import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class_name_map = {
    "Tomato_healthy": "Tomato Healthy",
    "Potato___healthy": "Potato: Healthy",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato: Tomato Yellow Leaf Curl Virus",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato Spider Mites Two Spotted Spider Mite",
    "Tomato_Leaf_Mold": "Tomato Leaf Mold",
    "Potato___Early_blight": "Potato: Early Blight",
    "Tomato__Tomato_mosaic_virus": "Tomato: Tomato Mosaic Virus",
    "Tomato_Early_blight": "Tomato Early Blight",
    "Potato___Late_blight": "Potato: Late Blight",
    "Pepper__bell___healthy": "Pepper: Bell: Healthy",
    "Tomato__Target_Spot": "Tomato: Target Spot",
    "Pepper__bell___Bacterial_spot": "Pepper: Bell: Bacterial Spot",
    "Tomato_Late_blight": "Tomato Late Blight",
    "Tomato_Bacterial_spot": "Tomato Bacterial Spot",
    "Tomato_Septoria_leaf_spot": "Tomato Septoria Leaf Spot"
}

def prepare_dataset(source_dir, output_base):
    splits = ['train', 'val', 'test']
    split_ratios = [0.7, 0.15, 0.15]

    for split in splits:
        os.makedirs(os.path.join(output_base, split), exist_ok=True)

    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        images = os.listdir(class_dir)
        random.shuffle(images)
        total = len(images)
        train_cut = int(total * split_ratios[0])
        val_cut = train_cut + int(total * split_ratios[1])

        split_data = {
            'train': images[:train_cut],
            'val': images[train_cut:val_cut],
            'test': images[val_cut:]
        }

        for split in splits:
            split_class_dir = os.path.join(output_base, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for img_name in split_data[split]:
                src = os.path.join(class_dir, img_name)
                dst = os.path.join(split_class_dir, img_name)
                shutil.copy2(src, dst)
            print(f"✅ {class_name}: {len(split_data['train'])} train, {len(split_data['val'])} val, {len(split_data['test'])} test")

def load_data(train_dir, val_dir, test_dir, image_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        zoom_range=0.2,
        rotation_range=15,
        shear_range=0.1
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )

    return train_generator, val_generator, test_generator
