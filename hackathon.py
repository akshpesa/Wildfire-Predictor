# Import necessary libraries
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the data
def load_data(data_dir, image_size=(64, 64)):
    images = []
    labels = []

    # Iterate through each class folder
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            for image_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, image_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)

# Load the dataset
data_dir = 'C:/Users/akshi/Downloads/hackIIITH/test'  # Update this with your dataset path
X, y = load_data(data_dir)

# Flatten the images
X = X.reshape(X.shape[0], -1)  # Flatten to 1D

# Step 2: Encode labels and split the data
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 3: Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Make predictions and evaluate the model
y_pred = model.predict(X_test)

# Print classification report and confusion matrix
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print(confusion_matrix(y_test, y_pred))

# Optional: Visualize some predictions
def plot_predictions(X, y_true, y_pred, label_encoder):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X[i].reshape(64, 64, 3))  # Reshape to original image size
        plt.title(f'True: {label_encoder.classes_[y_true[i]]}\nPredicted: {label_encoder.classes_[y_pred[i]]}')
        plt.axis('off')
    plt.show()

# Plot predictions
plot_predictions(X_test, y_test, y_pred, label_encoder)
