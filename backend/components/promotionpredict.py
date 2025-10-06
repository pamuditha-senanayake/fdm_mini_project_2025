# backend/components/promotionpredict.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from fastapi import APIRouter
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("data.csv")
df = df[['Product_Category', 'Customer_Segment', 'Shipping_Method', 'Payment_Method',
         'Gender', 'Income', 'Total_Purchases']]

# Target: High Purchaser if purchases >= 3
df['High_Purchase'] = (df['Total_Purchases'] >= 3).astype(int)
df = df.drop(columns=['Total_Purchases'])

# -----------------------------
# Handle imbalance (add noise + oversample)
# -----------------------------
low = df[df['High_Purchase'] == 0]
high = df[df['High_Purchase'] == 1]

# Oversample the minority class
if len(low) < len(high):
    low = low.sample(len(high), replace=True, random_state=42)
elif len(high) < len(low):
    high = high.sample(len(low), replace=True, random_state=42)

df_balanced = pd.concat([low, high])

# Shuffle
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Add a small noise column to encourage diversity
df_balanced['Random_Noise'] = np.random.randn(len(df_balanced))

# -----------------------------
# Split features & target
# -----------------------------
X = df_balanced.drop(columns=['High_Purchase'])
y = df_balanced['High_Purchase']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

categorical_cols = ['Product_Category', 'Customer_Segment', 'Shipping_Method',
                    'Payment_Method', 'Gender', 'Income']

# Convert categorical columns to category dtype
category_maps = {}
for col in categorical_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')
    category_maps[col] = X_train[col].cat.categories.tolist()

# -----------------------------
# Train LightGBM model
# -----------------------------
model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=64,
    random_state=42,
    is_unbalance=False  # we already balanced manually
)
model.fit(X_train, y_train, categorical_feature=categorical_cols)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("✅ Model trained to classify High Purchasers")
print(f"Accuracy 2A: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1 Score: {f1*100:.2f}%")

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
                         req.payment_method, req.gender, req.income, 0.0]],  # noise placeholder
                       columns=X.columns)

    # Replace unseen categories with "Other"
    for col in categorical_cols:
        if row[col][0] not in category_maps[col]:
            row[col] = "Other"
        row[col] = pd.Categorical(row[col], categories=category_maps[col] + ["Other"])

    # Add random noise for prediction
    row['Random_Noise'] = np.random.randn(1)

    # Predict probabilities
    proba = model.predict_proba(row)[0][1]  # probability of High Purchaser
    pred = int(proba >= 0.5)

    if pred == 1:
        return {"recommendation": f"🟢 Likely High Purchaser (prob={proba:.2f}) → No urgent promotion needed."}
    else:
        return {"recommendation": f"🔴 Low Purchaser  → Consider promotion."}
