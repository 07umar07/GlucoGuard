import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import imblearn as imb
from sklearn.model_selection import train_test_split

# Diabetes Prediction with Neural Network
# Machine Learning Models
# Date Created: October 28, 2024
# By Umar Abdul Hakim Robbani

# 1. Data Featuring
df = pd.read_csv('/home/umar/Documents/Datasets/Diabetes_Datasets/diabetes.csv')
# print(df.isnull().sum())
# zero_counts = df.eq(0).sum()
# print(zero_counts)

# Replace all zeros in each feature to the mean in each feature
df['Glucose'] = df['Glucose'].replace(0, np.median(df['Glucose']))
df['BloodPressure'] = df['BloodPressure'].replace(0, np.median(df['BloodPressure']))
df['SkinThickness'] = df['SkinThickness'].replace(0, np.median(df['SkinThickness']))
df['Insulin'] = df['Insulin'].replace(0, np.median(df['Insulin']))
df['BMI'] = df['BMI'].replace(0, np.median(df['BMI']))

# I decide to use features: Pregnancies, Glucose, Blood Pressure, BMI, DiabetesPedigreeFunction, Age

X = df.drop(['Outcome', 'SkinThickness', 'Insulin'], axis = 1).values
y = df['Outcome'].values

#2. Split Datasets
X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=0.25, random_state=42) # 0.176

data_shapes = {
  "Dataset": ["X_train", "X_dev", "X_test"],
  "Shape": [X_train.shape, X_dev.shape, X_test.shape],
  "Percentage": [100 / X.shape[0] * X_train.shape[0], 100 / X.shape[0] * X_dev.shape[0], 100 / X.shape[0] * X_test.shape[0]]
}
# print(X[0])

# #Save Train/Dev/Test to csv again for further analysis
# # Convert the split datasets back to DataFrames and add outcomes
# # Train set
# X_train_df = pd.DataFrame(X_train, columns=df.drop(['Outcome', 'SkinThickness', 'Insulin'], axis=1).columns)
# X_train_df['Outcome'] = y_train

# # Dev set
# X_dev_df = pd.DataFrame(X_dev, columns=df.drop(['Outcome', 'SkinThickness', 'Insulin'], axis=1).columns)
# X_dev_df['Outcome'] = y_dev

# # Test set
# X_test_df = pd.DataFrame(X_test, columns=df.drop(['Outcome', 'SkinThickness', 'Insulin'], axis=1).columns)
# X_test_df['Outcome'] = y_test

# # Save each DataFrame as a CSV file
# X_train_df.to_csv('train_data.csv', index=False)
# X_dev_df.to_csv('dev_data.csv', index=False)
# X_test_df.to_csv('test_data.csv', index=False)

# print("Data saved to CSV files.")

#3. Normalize the Data
# Calculate mean and standard deviation for training data
mean = np.mean(X_train, axis=0, keepdims= True)
std = np.std(X_train, axis=0, keepdims= True)

# Normalize training data
X_train = (X_train - mean) / std

# Normalize dev data using training data's mean and std
X_dev = (X_dev - mean) / std

# Normalize test data using training data's mean and std
X_test = (X_test - mean) / std

# 4. Checking whether the data is imbalanced or not
# print(np.bincount(y_train))
ROS = imb.over_sampling.RandomOverSampler()
# Data is imbalanced, so i decide to resample it 
X_train, y_train = ROS.fit_resample(X_train, y_train)


# # SAVING

# # Assuming you have preprocessed your data and split it into train/dev/test
# # Save necessary data into numpy arrays
# np.save('X_train.npy', X_train)
# np.save('X_dev.npy', X_dev)
# np.save('X_test.npy', X_test)
# np.save('y_train.npy', y_train)
# np.save('y_dev.npy', y_dev)
# np.save('y_test.npy', y_test)
# np.save('mean.npy', mean)
# np.save('std.npy', std)




