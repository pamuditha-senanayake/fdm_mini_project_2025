# components/segmentpredict.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "segment_model.joblib")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "segment_label_encoder.joblib")
DATA_PATH = os.path.join(BASE_DIR, "..", "data.csv")

# Globals
rf_classifier = None
label_encoders = {}
target_encoder = None


def load_and_train_segment_model():
    """Load or train the RandomForest model for customer segmentation."""
    global rf_classifier, label_encoders, target_encoder

    if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
        rf_classifier = joblib.load(MODEL_PATH)
        target_encoder = joblib.load(LABEL_ENCODER_PATH)
        return

    df = pd.read_csv(DATA_PATH)
    df.drop(columns=['Date', 'Year', 'Month', 'Time'], inplace=True, errors="ignore")

    X = df[['Age', 'Income', 'Total_Purchases', 'Amount']]
    y = df['Customer_Segment']

    # Encode categorical feature(s)
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    # Train RandomForest
    rf_classifier = RandomForestClassifier(
        n_estimators=100,  # reduce from 500 → 100
        max_depth=10,  # limit depth
        min_samples_split=5,  # avoid tiny branches
        random_state=42,
        class_weight="balanced",
        n_jobs=-1  # use parallel cores
    )

    rf_classifier.fit(X_train, y_train)

    # Save model + encoder
    joblib.dump(rf_classifier, MODEL_PATH)
    joblib.dump(target_encoder, LABEL_ENCODER_PATH)


def predict_segment(age: int, income: str, total_purchases: int, amount: float):
    """Predict customer segment given input values."""
    if rf_classifier is None or target_encoder is None:
        load_and_train_segment_model()

    input_df = pd.DataFrame([{
        "Age": int(age),
        "Income": str(income),
        "Total_Purchases": int(total_purchases),
        "Amount": float(amount)
    }])

    # Encode categorical income
    if "Income" in input_df.columns:
        le = LabelEncoder()
        input_df["Income"] = le.fit_transform(input_df["Income"])

    # Prediction
    pred_idx = rf_classifier.predict(input_df)[0]
    pred_label = target_encoder.inverse_transform([pred_idx])[0]

    # Probabilities
    probas = rf_classifier.predict_proba(input_df)[0]
    prob_dict = dict(zip(target_encoder.classes_, probas))

    # 🔑 Clean probabilities (NaN/inf → 0.0, cast to float)
    safe_prob_dict = {
        str(k): (0.0 if (v is None or np.isnan(v) or np.isinf(v)) else float(v))
        for k, v in prob_dict.items()
    }

    # Recommendation
    if pred_label == "Premium":
        recommendation = "Offer VIP perks, early access, and premium bundles."
    elif pred_label == "Regular":
        recommendation = "Encourage subscriptions and bundles."
    else:
        recommendation = "Use reactivation offers and reminders."

    return {
        "predicted_segment": pred_label,
        "probabilities": safe_prob_dict,
        "recommendation": recommendation
    }
if __name__ == "__main__":
    load_and_train_segment_model()
    print("✅ Segment model trained and saved successfully.")