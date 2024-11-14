import numpy as np
import tkinter as tk
# import matplotlib.pyplot as plt
from tflite_runtime.interpreter import Interpreter
# from sklearn.metrics import roc_curve, roc_auc_score
# from initialization import X_train, X_dev, X_test, y_train, y_dev, y_test, mean, std

# Load the TFLite model and allocate tensors
interpreter = Interpreter(model_path="diabetes_detector.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_tflite(interpreter, X):
    input_data = np.array(X, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction

# Load mean and std
mean = np.load("mean.npy")
std = np.load("std.npy")

def values():
    get_val = []
    for text in text_boxes:
        value = text.get()
        if value == '':
            value = 0
        get_val.append(float(value))
    X = np.array([get_val], dtype=np.float32)  # Explicitly set dtype to float32
    X = (X - mean) / std
    prediction = predict_tflite(interpreter, X)[0][0]
    print(prediction)

    # Shows the result
    if prediction >= 0.5:
        result = "A high chance you got diabetes, you should check your \nmedical condition ASAP."
    else:
        result = "No diabetes detected"

    result_label.config(text=f"Prediction: {result}\nChances: {prediction * 100:.2f}%")



# Creating the UI
# Main Window
root = tk.Tk()
root.title("Diabetes Detector")
root.geometry("370x400")

# Create caller
text_boxes = []
def label_and_text(labelText):
    label = tk.Label(root, text=labelText)
    label.pack()
    text_box = tk.Entry(root, width=15, justify='center')
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

chance_label = tk.Label(root, text="")
chance_label.pack()

submit_button = tk.Button(root, text="Submit", command=values)
submit_button.pack()

text_label = tk.Label(root, text="GlucoGuard v0.0\nBy Umar A.H. Robbani", font=('Helvetica', 8), justify='left')
text_label.place(relx=0.0, rely=1.0, anchor='sw')

# Start the GUI
root.mainloop()
