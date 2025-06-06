## Title
Oral Cancer Image Classification with Deep Learning and Machine Learning Models


## Description

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

## Dataset Information

This project uses two publicly available datasets:

1. **Kaggle Oral Cancer Image Dataset**  
   - Source: Kaggle (https://www.kaggle.com/datasets/zaidpy/oral-cancer-dataset)
   - Label structure: Binary (Cancer / No Cancer)
   - Format: JPEG images
   - Total Images: ~1000

2. **Roboflow Oral Cancer Dataset**  
   - Source: Roboflow (https://roboflow.com/](https://universe.roboflow.com/sagari-vijay/oral-cancer-data.)
   - Label structure: Binary (Cancer / No Cancer)
   - Format: PNG images
   - Total Images: ~1700

All datasets were preprocessed with resizing (224x224), normalization, and augmentation (horizontal flip, rotation, zoom).

## How to Run

Each notebook is self-contained and can be executed in order depending on the dataset being used. It is recommended to run the preprocessing notebooks (`01- Preprocesado_Kaggle.ipynb` and `01- Preprocesado_RoboFlow.ipynb`) before executing the main classification notebook (`ORAL_CANCER_CLASSIFICATION.ipynb`).

## Code Information

- Preprocessing is handled by two Jupyter notebooks for each dataset.
- The main classification pipeline (CNNs and Hybrid models) is in `ORAL_CANCER_CLASSIFICATION.ipynb`.
- All scripts are written in Python 3.8 using TensorFlow and scikit-learn.
- Models include DenseNet201, Inception-v3, and SVM, RF classifiers.

The code is modular and allows for adaptation to new datasets by changing file paths and label encodings.

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
pip install -r requirements.txt

## Methodology

- CNN architectures were fine-tuned using transfer learning.
- For hybrid models, deep features from the penultimate CNN layer were extracted and classified using ML models (SVM, Random Forest)
- Evaluation metrics: Accuracy, Sensitivity, Specificity, ROC AUC, F1-Score, and Diagnostic Odds Ratio (DOR).
- Cross-validation was conducted using 5-fold CV with stratified sampling.
- Statistical significance tested with Student's t-test and Wilcoxon signed-rank test.

## Citations

If you use this dataset or code, please cite:

Ormeño-Arriagada, P. et al. “Comparison of Deep Learning and Machine Learning Architectures for Early Oral Cancer Diagnosis,” *PeerJ Computer Science*, 2025.

## License

This project is shared under the MIT License. You are free to use, modify, and distribute it, provided that proper attribution is given.

## Contribution Guidelines

For feedback, suggestions, or improvements, please open an issue or submit a pull request.
