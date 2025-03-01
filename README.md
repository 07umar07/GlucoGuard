# Diabetes Prediction with Neural Network

## Overview
This project implements a neural network model to predict diabetes using a dataset containing various health indicators. The model is built using TensorFlow and Keras and is trained with techniques like data normalization, oversampling, and dropout regularization.

## Dataset
The dataset is sourced from the `diabetes.csv` file stored in Google Drive. It consists of various features, including:
- Glucose
- Blood Pressure
- BMI
- Skin Thickness
- Insulin
- Outcome (Target variable: 1 for diabetes, 0 for no diabetes)

## Data Preprocessing
1. Missing values and zero values in critical columns (Glucose, Blood Pressure, Skin Thickness, Insulin, BMI) are replaced with their median values.
2. The `SkinThickness` and `Insulin` columns are dropped from the input features.
3. Data is split into training, validation (dev), and test sets (70%-15%-15%).
4. The dataset is normalized using the mean and standard deviation of the training set.
5. Imbalanced classes are handled using `RandomOverSampler` from `imbalanced-learn`.

## Model Architecture
A deep neural network (DNN) is built using TensorFlow's Keras API with the following layers:
- Input layer with 6 features
- Multiple dense layers with ReLU activation and L2 regularization
- Batch normalization layers to improve training stability
- Dropout layers to prevent overfitting
- Final dense layer with a sigmoid activation function for binary classification

## Training
- Optimizer: Adam (learning rate = 0.0001)
- Loss function: Binary cross-entropy
- Batch size: 16
- Epochs: 100 (with early stopping)
- Callbacks:
  - EarlyStopping (monitors validation loss, patience = 10, restores best weights)
  - ReduceLROnPlateau (reduces learning rate if validation loss stops improving)

## Performance Evaluation
The trained model is evaluated on:
- Training set
- Development set
- Test set

Metrics: Accuracy is calculated for each dataset split.

## Model Saving
The trained model is saved in Google Drive:
```python
model.save('/content/drive/My Drive/Ml_Models_tensorflow/diabetes_det_gd2_51.keras')
```

## Dependencies
Install required libraries using:
```bash
pip install -r requirements.txt
```

## Running the Project
1. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. Load and preprocess the dataset.
3. Train the model.
4. Evaluate and save the trained model.

## Author
**Umar Abdul Hakim Robbani**  
**Date Created:** October 28, 2024

## Acknowledgments
This project was implemented in Google Colab using TensorFlow and Keras for deep learning.

