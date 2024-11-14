import numpy as np
import tkinter as tk
# from sklearn.metrics import roc_curve, roc_auc_score

# # Load your model
# model = tf.keras.models.load_model("/home/umar/myenv/SelfProject/Diabetes_Tensorflow/diabetes_det_gd2_4.keras")

# # Extract and save weights and biases
# dense_count = 0
# bn_count = 0

# for layer in model.layers:
#     if "batch_normalization" in layer.name:
#         np.save(f"bn_mean_{bn_count}.npy", layer.moving_mean.numpy())
#         np.save(f"bn_variance_{bn_count}.npy", layer.moving_variance.numpy())
#         bn_count += 1

# Number of layers
NUM_LAYERS_DENSE = 8
NUM_LAYERS_BN = 5

mean = np.load('mean.npy')
std = np.load('std.npy')

# Load weights and biases for dense layers
weights_np = [np.load(f"/home/umar/myenv/SelfProject/Diabetes_Tensorflow/Weight/weights_{i}.npy", allow_pickle=True) for i in range(NUM_LAYERS_DENSE)]
biases_np = [np.load(f"/home/umar/myenv/SelfProject/Diabetes_Tensorflow/Weight/biases_{i}.npy", allow_pickle=True) for i in range(NUM_LAYERS_DENSE)]

# Load batch norm parameters
bn_means = [np.load(f"/home/umar/myenv/SelfProject/Diabetes_Tensorflow/Weight/bn_mean_{i}.npy", allow_pickle=True) for i in range(NUM_LAYERS_BN)]
bn_variances = [np.load(f"/home/umar/myenv/SelfProject/Diabetes_Tensorflow/Weight/bn_variance_{i}.npy", allow_pickle=True) for i in range(NUM_LAYERS_BN)]

# Functions for activation and batch normalization
def batch_norm(x, mean, variance, epsilon=1e-5):
    return (x - mean) / np.sqrt(variance + epsilon)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# Function to perform forward pass using NumPy
def predict_np(X):
    # Normalize X
    # X = (X - mean) / std
    
    # First layer (Dense + Batch Norm)
    z1 = np.dot(X, weights_np[0]) + biases_np[0]
    a1 = relu(batch_norm(z1, bn_means[0], bn_variances[0]))
    
    # Second layer (Dense + Batch Norm)
    z2 = np.dot(a1, weights_np[1]) + biases_np[1]
    a2 = relu(batch_norm(z2, bn_means[1], bn_variances[1]))
    
    # Third layer (Dense + Batch Norm)
    z3 = np.dot(a2, weights_np[2]) + biases_np[2]
    a3 = relu(batch_norm(z3, bn_means[2], bn_variances[2]))
    
    # Fourth layer (Dense + Batch Norm)
    z4 = np.dot(a3, weights_np[3]) + biases_np[3]
    a4 = relu(batch_norm(z4, bn_means[3], bn_variances[3]))
    
    # Fifth layer (Dense, no Batch Norm)
    z5 = np.dot(a4, weights_np[4]) + biases_np[4]
    a5 = relu(z5)
    
    # Sixth layer (Dense + Batch Norm)
    z6 = np.dot(a5, weights_np[5]) + biases_np[5]
    a6 = relu(batch_norm(z6, bn_means[4], bn_variances[4]))
    
    # Seventh layer (Dense, no Batch Norm)
    z7 = np.dot(a6, weights_np[6]) + biases_np[6]
    a7 = relu(z7)
    
    # Output layer (Dense, no Batch Norm)
    z8 = np.dot(a7, weights_np[7]) + biases_np[7]
    output = sigmoid(z8)
    
    return output

# # Example of evaluating on test set using NumPy
# y_test_prediction_np = predict_np(X_test)
# ROCScore_np = roc_auc_score(y_test, y_test_prediction_np)
# print(f"ROC - AUC Score (NumPy): {ROCScore_np}")

# Creating the UI
# Main Window
root = tk.Tk()
root.title("Diabetes Detector")
root.geometry("370x400")

# Create caller
text_boxes = []
def label_and_text(labelText):
  label = tk.Label(root, text = labelText)
  label.pack()
  text_box = tk.Entry(root, width = 15, justify= 'center')
  text_box.pack()
  text_boxes.append(text_box)

label_and_text("Pregnancies:")
label_and_text("Glucose(mg/dL):")
label_and_text("Blood Pressure(mmHg):")
label_and_text("BMI(kg/m2):")
label_and_text("Diabetes Pedigree Function:")
label_and_text("Age:")

# Create values caller and show predictions
result_label = tk.Label(root, text="", font=("Helvetica", 9), justify='left')
result_label.pack()

chance_label = tk.Label(root, text= "")
chance_label.pack()

def values():
  get_val = []
  for text in text_boxes:
    value = text.get()
    if value == '':
      value = 0
    get_val.append(float(value))
  X = np.array([get_val])
  X = (X - mean) / std
  prediction = predict_np(X)[0][0]
  print(prediction)
  
  # Show the result
  if prediction >= 0.5 :
    result = "A high chance you got diabetes, you should check your \nmedical condition ASAP."
  else:
    result = "No diabetes detected"

  result_label.config(text = f"Prediction: {result}\nChances: {prediction * 100:.2f} %")

submit_button = tk.Button(root, text= "Submit", command= values)
submit_button.pack()

text_label = tk.Label(root, text="GlucoGuardian v0.0\nBy Umar A.H. Robbani", font= ('Helvetica', 8), justify= 'left')
text_label.place(relx=0.0, rely=1.0, anchor='sw')

# Start the GUI
root.mainloop()

