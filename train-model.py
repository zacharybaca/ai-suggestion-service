import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from scipy.sparse import hstack

# Load dataset
df = pd.read_csv("data.csv")

# Convert `taskCompleted` to binary (0 = Incomplete, 1 = Complete)
df["taskCompleted"] = df["taskCompleted"].astype(int)

# Vectorize text fields (`taskTitle` and `taskDetails`)
vectorizer = TfidfVectorizer()
X_title = vectorizer.fit_transform(df["taskTitle"])
X_details = vectorizer.fit_transform(df["taskDetails"])

# Encode `taskTodos` as the count of todo items
df["taskTodos"] = df["taskTodos"].apply(lambda x: len(str(x).split(',')))

# Encode assigned employee (target variable)
developer_encoder = LabelEncoder()
y = developer_encoder.fit_transform(df["assignedEmployee"])

# Combine all features
X = hstack((X_title, X_details, df["taskCompleted"].values.reshape(-1, 1), df["taskTodos"].values.reshape(-1, 1)))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save trained model and encoders
joblib.dump(model, "bug_assignment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(developer_encoder, "developer_encoder.pkl")

print("Model retrained and saved successfully!")
