import pandas as pd
import numpy as np
import lightgbm as lgb
from fastapi import APIRouter
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error

router = APIRouter()

model = None
category_maps = {}
X_columns = []
categorical_cols = ['Product_Category', 'Customer_Segment', 'Shipping_Method',
                    'Payment_Method', 'Gender', 'Income']


class PromotionRequest(BaseModel):
    product_category: str
    customer_segment: str
    shipping_method: str
    payment_method: str
    gender: str
    income: str


def train_promotion_model():
    global model, category_maps, X_columns

    df = pd.read_csv("data.csv")
    df = df[['Product_Category', 'Customer_Segment', 'Shipping_Method', 'Payment_Method',
             'Gender', 'Income', 'Total_Purchases']]
    df['High_Purchase'] = (df['Total_Purchases'] >= 3).astype(int)
    df = df.drop(columns=['Total_Purchases'])

    low = df[df['High_Purchase'] == 0]
    high = df[df['High_Purchase'] == 1]
    if len(low) < len(high):
        low = low.sample(len(high), replace=True, random_state=42)
    elif len(high) < len(low):
        high = high.sample(len(low), replace=True, random_state=42)
    df_balanced = pd.concat([low, high]).sample(frac=1, random_state=42).reset_index(drop=True)

    df_balanced['Random_Noise'] = np.random.randn(len(df_balanced))

    X = df_balanced.drop(columns=['High_Purchase'])
    y = df_balanced['High_Purchase']
    X_columns = X.columns.tolist()

    for col in categorical_cols:
        X[col] = X[col].astype('category')
        category_maps[col] = X[col].cat.categories.tolist()
        X[col] = X[col].cat.codes

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=12,
        num_leaves=64,
        min_data_in_leaf=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        reg_alpha=0.1,
        reg_lambda=0.1,
        is_unbalance=True,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        categorical_feature=[X.columns.get_loc(c) for c in categorical_cols],
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss',
        callbacks=[lgb.early_stopping(stopping_rounds=50)],

    )

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    acc = accuracy_score(y_val, y_pred) * 100

    print("\nMetrics for : Sales Forecast Dashboard")
    print("=== Press Detected for Forecast ===\n")
    print(f"MAE          : {mae:.2f}")
    print(f"RMSE         : {rmse:.2f}")
    print(f"Accuracy (%) : {acc:.2f}")
    print("\n===============================\n")


@router.post("/promotion")
def predict_purchase(req: PromotionRequest):
    global model

    if model is None:
        train_promotion_model()

    row = pd.DataFrame([[req.product_category, req.customer_segment, req.shipping_method,
                         req.payment_method, req.gender, req.income, 0.0]],
                       columns=X_columns)

    for col in categorical_cols:
        if row[col][0] not in category_maps[col]:
            row[col] = "Other"
            category_maps[col].append("Other")
        row[col] = pd.Categorical(row[col], categories=category_maps[col]).codes

    row['Random_Noise'] = np.random.randn(1)

    proba = model.predict_proba(row)[0][1]
    pred = int(proba >= 0.5)

    if pred == 1:
        return {"recommendation": f"🟢 Likely High Purchaser (prob={proba:.2f}) → No urgent promotion needed."}
    else:
        return {"recommendation": f"🔴 Low Purchaser → Consider promotion."}
