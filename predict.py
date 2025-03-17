from flask import Flask, request, jsonify
import joblib
import pandas as pd
import scipy.sparse

# Load the trained model and encoders
model = joblib.load("bug_assignment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
category_encoder = joblib.load("category_encoder.pkl")
developer_encoder = joblib.load("developer_encoder.pkl")

app = Flask(__name__)

@app.route('/predict-assignee', methods=['POST'])
def predict():
    data = request.json
    bug_description = data.get("description", "")
    category = data.get("category", "")

    # Convert input to numerical format
    X_text = vectorizer.transform([bug_description])
    category_encoded = category_encoder.transform([category]).reshape(-1, 1)
    X = scipy.sparse.hstack((X_text, category_encoded))

    # Make prediction
    prediction = model.predict(X)
    assigned_developer = developer_encoder.inverse_transform(prediction)[0]

    return jsonify({"assigned_developer": assigned_developer})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
