# PEERJ_OC_Classification
In this repository we include the code, and instructions for the reviewer of the paper in PEERJ 


# Oral Cancer Image Classification with Deep Learning and Machine Learning Models

This repository contains code and workflows used in the study titled **"Comparison of Deep Learning and Machine Learning Architectures for Early Oral Cancer Diagnosis"**, submitted to *PeerJ Computer Science* under the category **AI Applications**.

The study investigates and compares the performance of various convolutional neural network (CNN) architectures and hybrid classification approaches for oral cancer image classification, using two publicly available datasets. The provided notebooks include preprocessing steps, model training, evaluation, and result visualization.

## Repository Structure

### 1. `01- Preprocesado_Kaggle.ipynb`
This notebook handles the preprocessing of the **Kaggle Oral Cancer Image Dataset**, which includes:
- Dataset loading and inspection
- Class balancing procedures (e.g., oversampling)
- Image resizing and normalization
- Data splitting for training, validation, and testing
- Label encoding and saving of processed data for subsequent model training

### 2. `01- Preprocesado_RoboFlow.ipynb`
This notebook performs similar preprocessing for the **Roboflow Oral Cancer Image Dataset**, with:
- Image format conversion
- Data cleaning and augmentation
- Label transformation and dataset exploration
- Export of cleaned datasets for downstream analysis

### 3. `ORAL_CANCER_CLASSIFICATION.ipynb`
This is the core notebook where the main classification experiments are performed. It includes:
- Training of CNN models (DenseNet201, Inception-v3)
- Extraction of features for hybrid models using traditional classifiers (SVM, Random Forest, etc.)
- Evaluation of models using metrics such as accuracy, sensitivity, specificity, ROC AUC, and F1-score
- Visualization of confusion matrices and ROC curves
- Comparison of model performance across datasets and configurations

## How to Run

Each notebook is self-contained and can be executed in order depending on the dataset being used. It is recommended to run the preprocessing notebooks (`01- Preprocesado_Kaggle.ipynb` and `01- Preprocesado_RoboFlow.ipynb`) before executing the main classification notebook (`ORAL_CANCER_CLASSIFICATION.ipynb`).

### Requirements

- Python 3.8+
- Jupyter Notebook
- TensorFlow / Keras
- scikit-learn
- pandas
- matplotlib
- seaborn
- OpenCV
- tqdm

To install dependencies:
```bash
pip install -r requirements.txt
