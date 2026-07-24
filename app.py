from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)
model = None

model_path = os.path.join("model", "crop_model.pkl")

if os.path.exists(model_path):
    model = joblib.load(model_path)


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- ABOUT ----------------
@app.route("/about")
def about():
    return render_template("about.html")


# ---------------- FIND YOUR CROP ----------------
@app.route("/findyourcrop")
def findyourcrop():
    return render_template("findyourcrop.html")


# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return "Model not found."

    try:
        values = [
            float(request.form["nitrogen"]),
            float(request.form["phosphorous"]),
            float(request.form["potassium"]),
            float(request.form["temperature"]),
            float(request.form["humidity"]),
            float(request.form["ph"]),
            float(request.form["rainfall"])
        ]

        prediction = model.predict([values])[0]

        probabilities = model.predict_proba([values])[0]
        classes = model.classes_

        top3 = np.argsort(probabilities)[-3:][::-1]

        labels = [classes[i] for i in top3]
        scores = [round(probabilities[i] * 100, 2) for i in top3]

        return render_template(
            "result.html",
            prediction=prediction,
            labels=labels,
            scores=scores,
            user_values=values
        )

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)