import tensorflow as tf
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from initialization import X_train, X_dev, X_test, y_train, y_dev, y_test, mean, std
from sklearn.metrics import roc_curve, roc_auc_score

# We load model that i trained from Google Colab
model = load_model("/home/umar/myenv/SelfProject/Diabetes_Tensorflow/diabetes_det_gd2_4.keras")

# Check the accuracy in test set
loss, accuracy = model.evaluate(X_test, y_test) 
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Check accuracy in train set
loss, accuracy = model.evaluate(X_train, y_train) 
print(f"Train Accuracy: {accuracy * 100:.2f}%")

# Check accuracy in dev set
loss, accuracy = model.evaluate(X_dev, y_dev) 
print(f"Dev Accuracy: {accuracy * 100:.2f}%")

# ROC Score
# Model Prediction
y_test_prediction = model.predict(X_test)
ROCScore = roc_auc_score(y_test, y_test_prediction)
print(f"ROC - AUC Score: {ROCScore}")

# Plot the ROC Score
fpr, tpr, treshold = roc_curve(y_test, y_test_prediction)
plt.plot(fpr, tpr, lw = 1.5, label = f"ROC Curve, area = {ROCScore:.2f}")
plt.plot([0,1], [0,1], color = 'navy', lw= 1.5, linestyle= '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc = 'lower right')
plt.show()

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

# Create values caller and shows the predictions
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
  prediction = model.predict(X)[0][0]
  print(prediction)
  
  # Shows the result
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
