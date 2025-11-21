# Diabetic Retinopathy Detection Using a Hybrid CNN (ResNet50 + DenseNet121)

This project implements an AI-based system for detecting Diabetic Retinopathy (DR) from retinal fundus images. The core of the system is a custom hybrid deep learning model that combines ResNet50 and DenseNet121 feature extractors. The project includes a Flask-based backend for model inference and a React (Vite) frontend for user interaction.

---

## 1. Project Overview

Diabetic Retinopathy is one of the leading causes of blindness. Early detection can significantly improve treatment effectiveness. This project uses deep learning to automatically classify retinal images into DR severity stages.

The model is trained on publicly available datasets such as APTOS/Kaggle. It achieves improved performance by combining the strengths of two state-of-the-art architectures.

---

## 2. Hybrid Model Architecture

### 2.1 Base Models

Two pretrained CNN models are used:

- ResNet50
- DenseNet121

Both models are loaded with ImageNet weights and used as feature extractors. Their final classification layers are removed so only the convolutional backbone is retained.

### 2.2 Feature Extraction

- ResNet50 produces a 2048-dimensional feature vector.
- DenseNet121 produces a 1024-dimensional feature vector.

Both outputs are flattened and concatenated into a single 3072-dimensional vector.

### 2.3 Classification Head

The combined 3072-dimensional feature vector is passed through a custom fully connected classifier consisting of:

- Linear layers
- Batch Normalization
- Dropout
- ReLU activations

Final output corresponds to 5 classes representing DR severity.

### 2.4 Why Combine ResNet50 and DenseNet121?

- ResNet50 captures high-level hierarchical patterns.
- DenseNet121 captures fine-grained local features with dense connectivity.
- Combining both provides richer representations and improves classification accuracy.

The hybrid approach reduces overfitting and increases sensitivity to early-stage DR features.

---
### Install dependencies

pip install flask torch torchvision pillow numpy opencv-python

### Run the backend
python app.py

### Run the Frontend (React + Vite)
npm install
npm run dev

### Prediction Workflow

User uploads a retinal fundus image through the frontend.
The image is sent to the backend (POST /predict).
Backend preprocesses the image and passes it through the hybrid CNN model.
Backend returns the predicted DR class.
Frontend displays the result.

