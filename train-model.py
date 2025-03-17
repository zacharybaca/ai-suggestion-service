import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from scipy.sparse import hstack

# Load dataset
df = pd.read_csv("data.csv")

# Ensure 'category' column exists, if not create it with default value "Unknown"
if "category" not in df.columns:
    df["category"] = "Unknown"  # Create the column with a default value

# Fill missing categories with "Unknown"
df["category"] = df["category"].fillna("Unknown")

# Convert `taskCompleted` to binary (0 = Incomplete, 1 = Complete)
df["taskCompleted"] = df["taskCompleted"].astype(int)

# Vectorize text fields (`taskTitle` and `taskDetails`)
vectorizer = TfidfVectorizer()
X_title = vectorizer.fit_transform(df["taskTitle"])
X_details = vectorizer.fit_transform(df["taskDetails"])

# Encode `taskTodos` as the count of todo items
df["taskTodos"] = df["taskTodos"].apply(lambda x: len(str(x).split(',')))

# Encode `category` column and add "Unknown" category
categories = df["category"].unique().tolist() + ["Unknown"]  # Ensure "Unknown" is part of the classes

category_encoder = LabelEncoder()
category_encoder.fit(categories)  # Fit encoder with all known categories
df["category_encoded"] = category_encoder.transform(df["category"])

# Encode assigned employee (target variable)
developer_encoder = LabelEncoder()
y = developer_encoder.fit_transform(df["assignedEmployee"])

# Combine all features
X = hstack((
    X_title,
    X_details,
    df["taskCompleted"].values.reshape(-1, 1),
    df["taskTodos"].values.reshape(-1, 1),
    df["category_encoded"].values.reshape(-1, 1)  # Include category encoding
))
