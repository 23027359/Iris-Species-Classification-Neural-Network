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

print("\nLevel 3 Task 3 - Neural Networks with Keras → COMPLETED!")
print("You've now finished all Level 3 tasks!")
print("Capture the loss/accuracy curves + confusion matrix for your submission.")


# In[1]:


# =============================================================================
# LEVEL 3 - TASK 3: NEURAL NETWORKS WITH TENSORFLOW/KERAS
# Codveda Machine Learning Internship - Completed in Google Colab
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

print("TensorFlow version:", tf.__version__)  # Should show 2.15+ 

# ──── 1. Load Iris dataset from uploaded file ───────────────────────────────
df = pd.read_csv('/content/1) iris.csv')  # ← Colab path after upload
print("✓ Iris loaded! Shape:", df.shape)

# Features & target
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = df['species']

# Encode & one-hot
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = to_categorical(y_encoded, num_classes=3)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y_onehot, test_size=0.3, random_state=42, stratify=y_encoded
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp.argmax(axis=1)
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ──── 2. Build Neural Network ───────────────────────────────────────────────
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ──── 3. Train ──────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_val, y_val),
    verbose=1
)

# ──── 4. Plots ──────────────────────────────────────────────────────────────
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curves')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Curves')
plt.legend()

plt.tight_layout()
plt.show()

# ──── 5. Evaluation ─────────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Neural Network')
plt.show()


# In[3]:


# =============================================================================
# LEVEL 3 - TASK 3: NEURAL NETWORKS WITH TENSORFLOW/KERAS
# Codveda Machine Learning Internship - Completed in Google Colab
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

print("TensorFlow version:", tf.__version__)  # Should show 2.15+ 

# ──── 1. Load Iris dataset from uploaded file ───────────────────────────────
df = pd.read_csv('/content/1) iris.csv')  # ← Colab path after upload
print("✓ Iris loaded! Shape:", df.shape)

# Features & target
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = df['species']

# Encode & one-hot
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = to_categorical(y_encoded, num_classes=3)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y_onehot, test_size=0.3, random_state=42, stratify=y_encoded
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp.argmax(axis=1)
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ──── 2. Build Neural Network ───────────────────────────────────────────────
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ──── 3. Train ──────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_val, y_val),
    verbose=1
)

# ──── 4. Plots ──────────────────────────────────────────────────────────────
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curves')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Curves')
plt.legend()

plt.tight_layout()
plt.show()

# ──── 5. Evaluation ─────────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Neural Network')
plt.show()


# In[4]:


# Level 3 Task 3 – Neural Networks with Keras (Local Windows Version)
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

# Print version info
print("TensorFlow version:", tf.__version__)

# 1. Load data using your specific local path
# The 'r' before the string handles the Windows backslashes
file_path = r"C:\Users\Sindi\OneDrive\Codveda Matchine Learning Internship\1) iris.csv"

try:
    df = pd.read_csv(file_path)
    print("✓ Iris loaded successfully! Shape:", df.shape)
except FileNotFoundError:
    print(f"ERROR: Could not find the file at {file_path}")
    print("Please check if the file name or folder path is correct.")

# 2. Prepare features & target
# We assume the columns are: sepal_length, sepal_width, petal_length, petal_width, species
X = df.iloc[:, 0:4].values  # Takes the first 4 columns
y = df.iloc[:, 4]           # Takes the last column (species)

# Encode species names to numbers (e.g., Setosa -> 0, Versicolor -> 1)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# One-Hot Encode the numbers (e.g., 0 -> [1, 0, 0])
y_onehot = to_categorical(y_encoded, num_classes=3)

# Scale features (Crucial for Neural Networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split into Train, Validation, and Test sets
# 70% Train, 15% Val, 15% Test
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y_onehot, test_size=0.3, random_state=42, stratify=y_encoded
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# 4. Build the Neural Network Model
model = models.Sequential([
    layers.Input(shape=(4,)),                          # 4 input features


# In[6]:


# Level 3 Task 3 – Neural Networks with Keras (Local Windows Version)
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

# Print version info
print("TensorFlow version:", tf.__version__)

# 1. Load data using your specific local path
# The 'r' before the string handles the Windows backslashes
file_path = r"C:\Users\Sindi\OneDrive\Codveda Matchine Learning Internship\1) iris.csv"

try:
    df = pd.read_csv(file_path)
    print("✓ Iris loaded successfully! Shape:", df.shape)
except FileNotFoundError:
    print(f"ERROR: Could not find the file at {file_path}")
    print("Please check if the file name or folder path is correct.")

# 2. Prepare features & target
# We assume the columns are: sepal_length, sepal_width, petal_length, petal_width, species
X = df.iloc[:, 0:4].values  # Takes the first 4 columns
y = df.iloc[:, 4]           # Takes the last column (species)

# Encode species names to numbers (e.g., Setosa -> 0, Versicolor -> 1)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# One-Hot Encode the numbers (e.g., 0 -> [1, 0, 0])
y_onehot = to_categorical(y_encoded, num_classes=3)

# Scale features (Crucial for Neural Networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split into Train, Validation, and Test sets
# 70% Train, 15% Val, 15% Test
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y_onehot, test_size=0.3, random_state=42, stratify=y_encoded
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# 4. Build the Neural Network Model
model = models.Sequential([
    layers.Input(shape=(4,)),                          # 4 input features
    layers.Dense(64, activation='relu'),               # Hidden Layer 1
    layers.Dropout(0.3),                               # Prevent overfitting
    layers.Dense(32, activation='relu'),               # Hidden Layer 2
    layers.Dropout(0.2),
    layers.Dense(3, activation='softmax')              # Output Layer (3 classes)
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 5. Train the model
print("\nStarting training...")
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=16,
                    validation_data=(X_val, y_val),
                    verbose=1)

# 6. Visualize Results
plt.figure(figsize=(14, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss (Error)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 7. Final Evaluation on Test Set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

# Generate Predictions for Confusion Matrix
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# 8. Confusion Matrix Heatmap
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Neural Network')
plt.ylabel('Actual Species')
plt.xlabel('Predicted Species')
plt.show()



# In[ ]:




