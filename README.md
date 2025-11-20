# 🐄 Cattle Breed Classification Using Hybrid MobileNetV2 + Custom CNN  
A deep-learning based image classification system that identifies **three cattle breeds** with high accuracy:

- 🐂 Gir  
- 🐄 Jersey  
- 🐃 Sahiwal  

This project uses **MobileNetV2 + Custom CNN layers + Fine-tuning** to achieve strong performance with **74–78% validation accuracy**.

---

## Live Demo
Try the model online: [ https://huggingface.co/spaces/gaurav5005/breedAi ]

---

# 📌 Features
✅ Pretrained MobileNetV2 (Imagenet)  
✅ Custom CNN layers for improved feature extraction  
✅ Label smoothing for stable training  
✅ Data augmentation (heavy)  
✅ Fine-tuning last 20 layers  
✅ Prediction function (OpenCV-compatible)  
✅ Final model saved as `.keras`

---

# 📂 Dataset Structure

Cattle Breeds/
├── Gir/
├── Jersey cattle/
└── Sahiwal/

---
## Dataset
- Total Images: **1500**  
- Classes (Breeds): **3**  
- Images per Class: **500**  

---
- Data is split into:
  - **Training:** 80%  
  - **Validation:** 20%

---

## Model Architecture
- **Base Model:** MobileNetV2 (pretrained on ImageNet)  
- **Custom CNN Layers:** Conv2D + MaxPooling + GlobalAveragePooling  
- **Dense Layers:** 256 units with Dropout 0.5  
- **Output Layer:** 3 units with softmax activation  

**Training Details:**
- Loss: Categorical Crossentropy with label smoothing 0.1  
- Optimizer: Adam  
- Learning Rate: 1e-4 (initial), 1e-5 (fine-tuning)  
- Epochs: 30 (initial) + 10 (fine-tuning)  
- Callbacks: EarlyStopping, ReduceLROnPlateau

---

## Training Performance
**Base + Fine-tuned Hybrid Model:**
- Training Accuracy: ~**88%**  
- Validation Accuracy: ~**77%**  
- Loss on Validation: ~**0.7261**  

---

## Results
Hybrid MobileNetV2 + Custom CNN model gives the best performance.

Can predict the breed of a cow from an image with high confidence.

Saved model can be used for deployment or converted to TensorFlow Lite for mobile applications.

---

## Screenshort
<img width="1594" height="652" alt="Screenshot 2025-11-20 194132" src="https://github.com/user-attachments/assets/6810e342-5579-480d-a4d9-907231960f93" />

---

<img width="1599" height="577" alt="Screenshot 2025-11-20 194333" src="https://github.com/user-attachments/assets/93c5c102-27fc-4cc4-a9c4-598e16a6f9e3" />

---

## Author

Gaurav Yadav
C.S.E | MERN Stack & AI Enthusiast


