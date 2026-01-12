Task 3: Neural Networks for Multi-Class Classification
📋 Project Overview
This project is part of the Level 3 Machine Learning Internship tasks. It demonstrates the application of Deep Learning using TensorFlow and Keras to classify the classic Iris flower dataset. The model is designed to distinguish between three species (Setosa, Versicolor, and Virginica) based on physical sepal and petal measurements.

🧠 Neural Network Architecture
The model utilizes a Sequential architecture designed for robust classification while preventing overfitting:

Input Layer: Accepts 4 numerical features (sepal/petal length and width).

Hidden Layer 1: A dense layer with 64 neurons using ReLU activation for non-linear feature learning.

Dropout Layer (0.3): Regularization technique that randomly deactivates 30% of neurons to improve generalization.

Hidden Layer 2: A dense layer with 32 neurons and ReLU activation to further refine patterns.

Dropout Layer (0.2): A second dropout step (20%) to ensure the model does not rely too heavily on specific neurons.

Output Layer: A dense layer with 3 neurons and Softmax activation, providing a probability distribution across the three flower species.

🛠️ Technical Workflow
1. Data Preprocessing
Target Encoding: Species names are converted to numerical values using LabelEncoder and then One-Hot Encoded for compatibility with categorical crossentropy loss.

Feature Scaling: Implemented StandardScaler to normalize input features, which is essential for stable and fast convergence in neural networks.

Data Splitting: The dataset is stratified and split into Training (70%), Validation (15%), and Testing (15%) sets to ensure thorough evaluation.

2. Training Configuration
Optimizer: The Adam optimizer was used for efficient gradient descent.

Loss Function: Categorical Crossentropy was selected for this multi-class problem.

Parameters: The model was trained for 50 epochs with a batch size of 16.

📈 Evaluation & Results
The model's performance is monitored and evaluated through several key visualizations:

Loss & Accuracy Curves: Training and validation plots are generated to ensure the model converged without significant overfitting.

Classification Report: Provides detailed Precision, Recall, and F1-scores for each flower species.

Confusion Matrix: A heatmap visualization showing the model's accuracy in predicting the actual species on the unseen test set.

💻 How to Run
Dependencies: Install required libraries:

Bash

pip install tensorflow pandas scikit-learn seaborn matplotlib
Dataset Path: Ensure the file 1) iris.csv is located at the path specified in the script or update the file_path variable.

Execution: Run NeuralNetworks.py to train the model and generate the evaluation plots.

Intern: Sindisiwe

Internship: Codveda Machine Learning

Task Status: Level 3 Task 3 - COMPLETED
