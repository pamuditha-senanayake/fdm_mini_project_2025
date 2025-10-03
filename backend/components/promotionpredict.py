# backend/components/promotionpredict.py
#https://colab.research.google.com/drive/14CqX81DjqB4hsiOaPriHE3Xqej7oNS4J?usp=sharing

import pandas as pd
import numpy as np
import lightgbm as lgb
from fastapi import APIRouter
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data.csv")
df = df[['Product_Category', 'Customer_Segment', 'Shipping_Method', 'Payment_Method',
         'Gender', 'Income', 'Total_Purchases']]

# Create classification target: High Purchaser = 1 if purchases >= 3
df['High_Purchase'] = (df['Total_Purchases'] >= 3).astype(int)
df = df.drop(columns=['Total_Purchases'])

# Split features & target
X = df.drop(columns=['High_Purchase'])
y = df['High_Purchase']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define categorical columns
categorical_cols = ['Product_Category', 'Customer_Segment', 'Shipping_Method', 'Payment_Method', 'Gender', 'Income']

# Convert categorical columns to category dtype and save categories
category_maps = {}
for col in categorical_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')
    category_maps[col] = X_train[col].cat.categories

# -----------------------------
# Train LightGBM model
# -----------------------------
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=64,
    random_state=42
)
model.fit(X_train, y_train, categorical_feature=categorical_cols)

# -----------------------------
# Evaluate (optional)
# -----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("✅ Model trained to classify High Purchasers")
print(
    f"Accuracy: {acc * 100:.2f}% | Precision: {prec * 100:.2f}% | Recall: {rec * 100:.2f}% | F1 Score: {f1 * 100:.2f}%")

# -----------------------------
# FastAPI router
# -----------------------------
router = APIRouter()


class PromotionRequest(BaseModel):
    product_category: str
    customer_segment: str
    shipping_method: str
    payment_method: str
    gender: str
    income: str


@router.post("/promotion")
def predict_purchase(req: PromotionRequest):
    # Create a single row DataFrame
    row = pd.DataFrame([[req.product_category, req.customer_segment, req.shipping_method,
                         req.payment_method, req.gender, req.income]],
                       columns=X.columns)

    # Ensure categories match training
    for col in categorical_cols:
        row[col] = pd.Categorical(row[col], categories=category_maps[col])

    # Predict
    pred = model.predict(row)[0]

    # Return human-readable recommendation
    if pred == 1:
        return {"recommendation": "🟢 Likely High Purchaser → No urgent promotion needed."}
    else:
        return {"recommendation": "🔴 Low Purchaser → Consider promotion."}
