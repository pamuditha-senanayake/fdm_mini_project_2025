import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal


# --- MODEL TRAINING AND SAVING (RUN THIS SECTION ONCE) ---

# This function simulates the user's notebook logic and saves the models/transformers
def train_and_save_artifacts():
    print("--- Running Training and Saving Artifacts (Run this once) ---")

    # 1. Create Dummy Data if not exists
    if not os.path.exists("data.csv"):
        data = {
            "Customer_ID": range(1, 101),
            "Age": np.random.randint(20, 60, 100),
            "Gender": np.random.choice(["Male", "Female", "Other"], 100),
            "Income": np.random.choice(["Low", "Medium", "High", np.nan, "Unknown"], 100, p=[0.2, 0.3, 0.3, 0.1, 0.1]),
            "Customer_Segment": np.random.choice(["Bronze", "Silver", "Gold", "Platinum"], 100),
            "Amount": np.random.uniform(10, 500, 100),
            "Total_Purchases": np.random.randint(1, 20, 100),
            "Product_Category": np.random.choice(["Electronics", "Clothing", "Home Goods", "Books", "Other"], 100),
            "Feedback": np.random.choice(["Good", "Bad", np.nan, "No Feedback"], 100)
        }
        df = pd.DataFrame(data)
        df.to_csv("data.csv", index=False)
        print("Created dummy data.csv")

    # 2. Data Loading and Preprocessing
    df = pd.read_csv("data.csv")
    df = df.drop_duplicates()
    df = df.dropna(subset=["Customer_ID", "Amount", "Product_Category"])
    df = df.fillna({"Income": "Unknown", "Feedback": "No Feedback"})

    cat_cols = ["Gender", "Income", "Customer_Segment", "Product_Category"]
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    joblib.dump(label_encoders, "label_encoders.joblib")

    num_cols = ["Age", "Amount", "Total_Purchases"]
    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])
    joblib.dump(imputer, "num_imputer.joblib")

    X = df[["Age", "Gender", "Income", "Customer_Segment", "Amount", "Total_Purchases"]]
    y = df["Product_Category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Model Training, Accuracy Check, and Saving
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5),
        "SVC": SVC(kernel="rbf", probability=True, random_state=42)
    }

    accuracies = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        accuracies[name] = acc
        joblib.dump(model, f"{name}.joblib")
        print(f"{name} trained with accuracy: {acc:.4f}")

    best_model = max(accuracies, key=accuracies.get)
    print(f"\nTraining complete. Best model is: {best_model}")
    return best_model


# --- FASTAPI SERVICE SETUP ---

try:
    # Attempt to load saved models and transformers
    models = {
        "LogisticRegression": joblib.load("LogisticRegression.joblib"),
        "RandomForestClassifier": joblib.load("RandomForestClassifier.joblib"),
        "GradientBoostingClassifier": joblib.load("GradientBoostingClassifier.joblib"),
        "KNeighborsClassifier": joblib.load("KNeighborsClassifier.joblib"),
        "SVC": joblib.load("SVC.joblib")
    }
    label_encoders = joblib.load("label_encoders.joblib")
    num_imputer = joblib.load("num_imputer.joblib")

    target_le = label_encoders["Product_Category"]
    category_mapping = {int(k): str(v) for k, v in
                        dict(zip(target_le.transform(target_le.classes_), target_le.classes_)).items()}
    best_model_name = "RandomForestClassifier"  # Set based on typical performance or result of `train_and_save_artifacts`

except FileNotFoundError:
    print("\n--- WARNING: Model artifacts not found. Running training now... ---")
    best_model_name = train_and_save_artifacts()
    # Reload the artifacts after training
    models = {
        "LogisticRegression": joblib.load("LogisticRegression.joblib"),
        "RandomForestClassifier": joblib.load("RandomForestClassifier.joblib"),
        "GradientBoostingClassifier": joblib.load("GradientBoostingClassifier.joblib"),
        "KNeighborsClassifier": joblib.load("KNeighborsClassifier.joblib"),
        "SVC": joblib.load("SVC.joblib")
    }
    label_encoders = joblib.load("label_encoders.joblib")
    num_imputer = joblib.load("num_imputer.joblib")
    target_le = label_encoders["Product_Category"]
    category_mapping = {int(k): str(v) for k, v in
                        dict(zip(target_le.transform(target_le.classes_), target_le.classes_)).items()}

app = FastAPI()


# Pydantic Schema
class PredictionInput(BaseModel):
    Age: int
    Gender: Literal["Male", "Female", "Other"]
    Income: Literal["Low", "Medium", "High", "Unknown"]
    Customer_Segment: Literal["Bronze", "Silver", "Gold", "Platinum"]
    Amount: float
    Total_Purchases: int
    model_name: Literal[
        "LogisticRegression",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "KNeighborsClassifier",
        "SVC"
    ] = best_model_name


# Preprocessing Utility
def preprocess_input(data: PredictionInput):
    df = pd.DataFrame([data.model_dump(exclude={"model_name"})])

    # Label Encode categorical features
    cat_cols = ["Gender", "Income", "Customer_Segment"]
    for col in cat_cols:
        le = label_encoders[col]
        # In a real app, you'd handle unseen labels, here we rely on Literal
        df[col] = le.transform(df[col].astype(str))

    # Impute numerical features
    num_cols = ["Age", "Amount", "Total_Purchases"]
    df[num_cols] = num_imputer.transform(df[num_cols])

    return df


# Prediction Endpoint
@app.post("/predict")
def predict_category(data: PredictionInput):
    model = models.get(data.model_name)
    if not model:
        raise HTTPException(status_code=400, detail=f"Model '{data.model_name}' not found.")

    processed_data = preprocess_input(data)
    prediction_encoded = model.predict(processed_data)[0]
    prediction_category = category_mapping.get(prediction_encoded, f"Encoded Category {prediction_encoded}")

    return {
        "model_used": data.model_name,
        "predicted_category": prediction_category,
        "best_model_suggestion": best_model_name
    }


@app.get("/models")
def get_models():
    return {
        "available_models": list(models.keys()),
        "best_model_suggestion": best_model_name,
        "best_model_note": "RandomForestClassifier typically performs best on this type of classification task due to its robustness and ensemble nature."
    }

# --- To Run the Backend ---
# 1. Save the code as backend.py
# 2. Execute the training logic if the files are not present.
# 3. Start the server: uvicorn backend:app --reload