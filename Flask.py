from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np


# Load dataset
df = pd.read_csv("toy_dataset.csv")
# read first few rows
print(df.head())

# replace
df = df.replace(
    to_replace=['Austin', "Boston", "Dallas", "Los Angeles", "Mountain View", "New York City", "San Diego", "Washington D.C."], 
    value=[1,2,3,4,5,6,7,8,])
#print(df.head())
#print(df.tail())

df = df.replace(
    to_replace=['Male', "Female"], 
    value=[0,1])

# Select independent and dependent vairables\
X = df[["City", "Gender", "Age", "Income"]]
y = df["Illness"]

# Splting the data into traing and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# model
classifier = RandomForestClassifier()

# fit model
classifier.fit(X_train, y_train)

# pickle
pickle.dump(classifier, open("model.pkl", "wb"))



app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    
    return render_template("index.html", prediction_text = "Is the preson Ill? (Yes or No)  {}".format(prediction))


if __name__ == '__main__':
    app.run(port = 3000, debug=True)