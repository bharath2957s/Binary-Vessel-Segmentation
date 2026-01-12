# Binary Vessel Segmentation using DRIVE Dataset 👁️

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20TensorFlow-orange)
![License](https://img.shields.io/badge/License-MIT-green)

This project implements **binary retinal blood vessel segmentation** using deep learning techniques on the **DRIVE (Digital Retinal Images for Vessel Extraction)** dataset. The goal is to classify each pixel in a retinal fundus image as either **vessel** or **background**.



## 📋 Table of Contents
- [Objectives](#-objectives)
- [Dataset](#-dataset)
- [Models & Architectures](#-models--architectures)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Evaluation Metrics](#-evaluation-metrics)

---

## 🎯 Objectives
The primary goals of this project are:
* Perform accurate binary vessel segmentation on retinal images.
* Train and evaluate deep learning segmentation models.
* Compare baseline architectures against improved variations.
* Visualize segmentation results interactively using a Streamlit demo application.

---

## 💾 Dataset
This project uses the **DRIVE (Digital Retinal Images for Vessel Extraction)** dataset.
* **Input:** RGB Retinal Fundus Images.
* **Output:** Binary Masks (White = Vessel, Black = Background).

---

## 🧠 Models & Architectures

We explore two main architectural approaches to solve the segmentation task:

### 1. Baseline Model
* **Architecture:** `2D U-Net`
* **Loss Function:** Dice Loss
* **Description:** A standard encoder-decoder network widely used for biomedical image segmentation.

### 2. Improved Model
* **Architecture:** `U-Net++` (Nested U-Net)
* **Loss Function:** Combined Loss (Binary Cross-Entropy + Dice Loss)
* **Description:** An improved architecture with nested skip pathways to reduce the semantic gap between the encoder and decoder sub-networks.

---

## 📂 Project Structure

```text
├── app2.py             # Streamlit demo application for visualization
├── dataset.py          # DRIVE dataset loader and preprocessing logic
├── evaluate.py         # Script for model evaluation and metrics calculation
├── model.py            # Definitions of U-Net and U-Net++ architectures
├── requirements.txt    # List of required Python libraries
└── train.py            # Main script for training the models
