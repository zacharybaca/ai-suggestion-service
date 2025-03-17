from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv  # type: ignore
import joblib
import pandas as pd
import scipy.sparse

# Load the trained model and encoders
model = joblib.load("bug_assignment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
category_encoder = joblib.load("category_encoder.pkl")
developer_encoder = joblib.load("developer_encoder.pkl")

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Get PORT from .env or default to 5000
port = os.getenv("PORT", 5000) 

@app.route('/predict-assignee', methods=['POST'])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid input"}), 400

    bug_description = data.get("description", "")
    category = data.get("category", "")

    if category not in category_encoder.classes_:
        category = "Unknown"  # Map unknown categories to "Unknown"

    # Convert input to numerical format
    X_text = vectorizer.transform([bug_description])
    category_encoded = category_encoder.transform([category]).reshape(-1, 1)
    X = scipy.sparse.hstack((X_text, category_encoded))

    # Make prediction
    prediction = model.predict(X)
    assigned_developer = developer_encoder.inverse_transform(prediction)[0]

    return jsonify({"assigned_developer": assigned_developer})  # Moved to the end

if __name__ == '__main__':
    app.run(port=port, debug=True)
