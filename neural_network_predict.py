
import pandas as pd
import keras
import numpy as np



dataset = pd.read_csv('predict_cancer.csv')

INPUT = list(dataset.columns[1:])
X = []
for i in INPUT:
    X.append(float(i))
    
X = np.array(X).reshape(1, -1)
print("inputs are",X)

classifier = keras.models.load_model("neural_network_model")

pred = classifier.predict(X)

if pred[0] == 0:
    print("prediction is : Benign")
else:
    print("prediction is : Malignant")

