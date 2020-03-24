
import pandas as pd
import pickle


dataset = pd.read_csv('predict_cancer.csv')

X = list(dataset.columns[1:])
input = []
input.append(X)
print("Input values are: ",input)


classifier = pickle.load(open("random_forest_model.pkl", "rb"))

pred = classifier.predict(input)

if pred[0] == 0:
    print("prediction is : Benign")
else:
    print("prediction is : Malignant")

