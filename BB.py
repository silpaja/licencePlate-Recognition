import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
from PIL import Image
import pandas as pd

# Function to verify and process images
def validate_and_resize_image(image_path, target_size=(224, 224)):
    try:
        with Image.open(image_path) as img:
            img.verify()  # Check for corruption
        with Image.open(image_path) as img:
            img = img.convert('RGB')  # Ensure valid mode
            img = img.resize(target_size)  # Resize to target size
            return np.array(img) / 255.0
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Function to load images and labels
def load_data(images_dir, labels_dir, image_size=(224, 224)):
    images = []
    labels = []

    for filename in os.listdir(images_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(images_dir, filename)
            label_path = os.path.join(labels_dir, filename.replace('.jpg', '.txt'))

            if os.path.exists(label_path):
                # Load and validate image
                img = validate_and_resize_image(image_path, target_size=image_size)
                if img is not None:
                    images.append(img)

                    # Load label
                    with open(label_path, 'r') as f:
                        label = f.read().strip().split('\n')
                        label = [list(map(float, item.split())) for item in label]
                        labels.append(label)
                else:
                    print(f"Skipping corrupted or invalid image: {image_path}")

    return np.array(images), labels

# Function to prepare labels
def prepare_labels(labels, image_size=(224, 224), max_boxes=10):
    prepared_labels = []

    for label in labels:
        prepared_label = []
        for box in label:
            if len(box) >= 5:
                class_id, x_center, y_center, width, height = box
                prepared_label.append([
                    class_id,
                    x_center * image_size[0],
                    y_center * image_size[1],
                    width * image_size[0],
                    height * image_size[1]
                ])
        while len(prepared_label) < max_boxes:
            prepared_label.append([0, 0, 0, 0, 0])
        prepared_labels.append(prepared_label[:max_boxes])

    return prepared_labels

# Function to pad or truncate predictions
def pad_or_truncate_boxes(boxes, max_boxes=10):
    padded_boxes = []
    for box in boxes:
        if len(box) < max_boxes * 5:
            box = np.pad(box, (0, max_boxes * 5 - len(box)), mode='constant')
        else:
            box = box[:max_boxes * 5]
        padded_boxes.append(box)
    return np.array(padded_boxes)

# Load training data
images_dir = '/content/train'
labels_dir = '/content/labels'
images, labels = load_data(images_dir, labels_dir)
prepared_labels = prepare_labels(labels, max_boxes=10)

# Simple model for demonstration
model = models.Sequential([
    layers.InputLayer(input_shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(50, activation='linear')  # Predict 10 boxes, each with 5 values
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Convert prepared labels to training format
train_labels = pad_or_truncate_boxes(
    [np.array(box).flatten() for box in prepared_labels], max_boxes=10
)

# Train the model
history = model.fit(images, train_labels, epochs=3, batch_size=1, validation_split=0.2)

# Load test data
test_images_dir = '/content/test'
test_labels_dir = '/content/testlabel'
test_images, test_labels = load_data(test_images_dir, test_labels_dir)
prepared_test_labels = prepare_labels(test_labels, max_boxes=10)

# Predict on test data
predictions = model.predict(test_images)
predicted_boxes = pad_or_truncate_boxes(predictions, max_boxes=10)

# Prepare predictions for Excel
df = pd.DataFrame({
    'Image_ID': [f"test_image_{i+1}" for i in range(len(predicted_boxes))],
    'Predicted_Boxes': list(predicted_boxes),
    'True_Labels': list(prepared_test_labels)
})

# Save predictions to Excel
df.to_excel('test_predictions.xlsx', index=False)
print("Predictions saved to test_predictions.xlsx")
