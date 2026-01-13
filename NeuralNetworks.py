#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =============================================================================
# LEVEL 3 - TASK 3: NEURAL NETWORKS WITH TENSORFLOW/KERAS
# Codveda Machine Learning Internship
# Dataset: Iris (1) iris.csv) → Multi-class classification (3 species)
# Objectives:
#   - Design a neural network architecture
#   - Train with backpropagation
#   - Visualize training/validation loss & accuracy curves
#   - Evaluate final accuracy
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ──── Visual style ──────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ──── 1. Load and prepare the Iris dataset ──────────────────────────────────
# Use your actual path here (adjust if needed)
iris_path = r"C:\Users\Sindi\OneDrive\Codveda Matchine Learning Internship\1) iris.csv"

try:
    df = pd.read_csv(iris_path)
    print("✓ Iris dataset loaded successfully!")
    print("Shape:", df.shape)
except FileNotFoundError:
    print("Iris file not found! Check path:", iris_path)
    raise

# Features & target
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = df['species']

# Encode target (setosa=0, versicolor=1, virginica=2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# One-hot encoding for neural network output
y_onehot = to_categorical(y_encoded, num_classes=3)

# Scale features (very important for neural nets!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train / Validation / Test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y_onehot, test_size=0.3, random_state=42, stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp.argmax(axis=1)
)

print(f"\nShapes:")
print(f"  Train: {X_train.shape} → {y_train.shape}")
print(f"  Val:   {X_val.shape}   → {y_val.shape}")
print(f"  Test:  {X_test.shape}  → {y_test.shape}\n")

# ──── 2. Build the Neural Network Architecture ──────────────────────────────
model = models.Sequential([
    layers.Input(shape=(4,)),                    # 4 input features
    layers.Dense(64, activation='relu'),         # Hidden layer 1
    layers.Dropout(0.3),                         # Dropout to prevent overfitting
    layers.Dense(32, activation='relu'),         # Hidden layer 2
    layers.Dropout(0.2),
    layers.Dense(3, activation='softmax')        # Output: 3 classes
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ──── 3. Train the model ────────────────────────────────────────────────────
print("\nTraining neural network...\n")

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_val, y_val),
    verbose=1
)

# ──── 4. Plot training history ──────────────────────────────────────────────
plt.figure(figsize=(14, 5))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# ──── 5. Evaluate on test set ───────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test Loss:     {test_loss:.4f}\n")

# Predictions & Classification Report
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Neural Network (Test Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()






# In[ ]:




