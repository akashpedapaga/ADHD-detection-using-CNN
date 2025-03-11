
# Explainable CNN-Based ADHD Detection using EEG Data

## Overview
This repository contains the implementation of an **Explainable AI (XAI) approach for ADHD detection** using **Convolutional Neural Networks (CNNs)** trained on **EEG data**. The model leverages **Local Interpretable Model-agnostic Explanations (LIME)** and **SHAPley Additive Explanations (SHAP)** to enhance interpretability.

## Features
- **EEG Data Processing:** Prepares EEG signals from ADHD and control subjects.
- **CNN Model:** Trained for ADHD classification based on EEG patterns.
- **Explainability Techniques:**
  - **LIME:** Provides local feature importance explanations.
  - **SHAP:** Offers a global perspective on feature contributions.
- **Performance Evaluation:** Includes **ROC Curves, Precision-Recall Curves, Confusion Matrix, and Training Metrics Visualization**.

## Project Structure
```
├── EEG/                         # Main directory containing EEG datasets
│   ├── ADHD Data Reduction/      # Preprocessed ADHD datasets
│   ├── Reduced Reduced Datasets/ # Further reduced datasets for training
│   ├── CSV Files & MATLAB Data  # Raw EEG data in CSV & MAT format
│   ├── Explanation Files/        # LIME & SHAP Explanation Results
│   ├── Figures/                  # Performance Evaluation Plots
│   ├── Final.py                  # Main script for ADHD detection & explanation
│   ├── Other Python Scripts      # Supporting code for analysis
│   ├── README.md                 # Project Documentation
```

## Main Script: `Final.py`
The core implementation of the project is contained in `Final.py`, which:
1. Loads and preprocesses EEG data.
2. Trains a **1D CNN model** for ADHD detection.
3. Uses **LIME and SHAP** for explainability.
4. Evaluates the model's performance.

## Requirements
To run this project, install the necessary dependencies using:
```sh
pip install -r requirements.txt
```
List of key dependencies:
- `tensorflow`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `lime`
- `imblearn`

## Running the Project
1. Clone the repository:
   ```sh
   git clone https://github.com/akashpedapaga/ADHD-detection-using-CNN.git
   cd ADHD-detection-using-CNN
   ```
2. Run the main script:
   ```sh
   python Final.py
   ```

## Results
The model achieves **98.91% accuracy** on unseen EEG data. Performance evaluation metrics:
- **ROC Curve & AUC Score**
- **Confusion Matrix**
- **Feature Importance Analysis using LIME & SHAP**
- **Training & Validation Loss/Accuracy Graphs**

## Acknowledgments
This research was conducted as part of **Akash Babu Pedapaga's Master's Thesis** at **Rutgers University – Camden**, under the guidance of:
- **Dr. Sheik Rabiul Islam**
- **Dr. Desmond Lun**


## License
This project is licensed under the MIT License.

---

For further inquiries, contact **akashpedapaga@gmail.com** or visit **[LinkedIn](https://www.linkedin.com/in/akash-babu-pedapaga/)**.
