from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
from src.components.data_transformation import preprocess_data
from src.pipeline.predict_pipeline import preprocess_form_data

app = Flask(__name__)

def load_model():
    with open('artifacts/model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form.to_dict()
        print(form_data)
        features = preprocess_form_data(form_data)
        print(features.columns)
        model = load_model()
        prediction = model.predict(features)
        # prediction = model.predict(processed_data)

        prediction_text = f"Predicted Flight Price: {round(prediction[0])}"
        return render_template("index.html", prediction_text=prediction_text)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)