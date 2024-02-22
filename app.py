from flask import Flask, request, render_template, jsonify
import pickle
from src.pipeline.predict_pipeline import FlightDataPreprocessor

app = Flask(__name__)
preprocessor = FlightDataPreprocessor()

def load_model(model_path='artifacts/model.pkl'):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form.to_dict()
        features = preprocessor.preprocess_form_data(form_data)
        model = load_model()
        prediction = model.predict(features)
        prediction_text = f"Predicted Flight Price: {round(prediction[0])}"
        return render_template("index.html", prediction_text=prediction_text)
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template("index.html", prediction_text=error_message)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)