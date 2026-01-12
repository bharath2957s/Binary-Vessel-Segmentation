# Binary Vessel Segmentation using DRIVE Dataset 👁️

This project implements **binary retinal blood vessel segmentation** using deep learning techniques on the **DRIVE (Digital Retinal Images for Vessel Extraction)** dataset.  
Each pixel in a retinal fundus image is classified as **vessel** or **background**.

---

## 🎯 Objectives
- Perform binary vessel segmentation on retinal images
- Train and evaluate deep learning segmentation models
- Compare baseline and improved architectures
- Visualize segmentation results using a demo application

---

## 🧠 Models Used
- **Baseline Model**
  - Architecture: 2D UNet
  - Loss Function: Dice Loss

- **Improved Model**
  - Architecture: UNet++
  - Loss Function: Combined Loss (Binary Cross-Entropy + Dice)

---

## 📂 Project Files
- `train.py` – Model training
- `evaluate.py` – Model evaluation and metrics
- `model.py` – Segmentation model architectures
- `dataset.py` – DRIVE dataset loader and preprocessing
- `app2.py` – Streamlit demo application
- `requirements.txt` – Required Python libraries

---

## 📊 Evaluation Metrics
- Dice Score  
- Accuracy  
- Precision  
- Recall (Sensitivity)  
- Specificity  
- ROC-AUC  
- PR-AUC  

---

## 🚀 How to Run the Project

### Install dependencies
```bash
pip install -r requirements.txt
