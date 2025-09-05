import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer

DATA_PATH = Path("data/raw/data.csv")
SAMPLE_FOR_DEV = 200_000
RANDOM_STATE = 42
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH, low_memory=False)

if SAMPLE_FOR_DEV and len(df) > SAMPLE_FOR_DEV:
    df = df.sample(SAMPLE_FOR_DEV, random_state=RANDOM_STATE).reset_index(drop=True)

def to_snake(name):
    return name.strip().replace(" ","_").replace("/","_").replace("-","_").lower()
df.columns = [to_snake(c) for c in df.columns]

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors='coerce')

for col in ["amount","total_amount","ratings","total_purchases","age"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

df = df.drop_duplicates().reset_index(drop=True)

if "date" in df.columns:
    df["year_num"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)

for col in ["amount","total_amount"]:
    if col in df.columns:
        q1,q99 = df[col].quantile([0.01,0.99])
        df[col+"_clipped"] = df[col].clip(q1,q99)

if {"customer_id","date","amount"}.issubset(df.columns):
    last_date = df["date"].max()
    cust_grp = df.groupby("customer_id").agg(
        last_purchase=("date","max"),
        frequency=("transaction_id","nunique") if "transaction_id" in df.columns else ("customer_id","count"),
        monetary=("amount","sum")
    )
    cust_grp["recency_days"] = (last_date - cust_grp["last_purchase"]).dt.days
    df = df.merge(cust_grp[["recency_days","frequency","monetary"]].reset_index(), on="customer_id", how="left")

if "product_brand" in df.columns:
    top_brands = df["product_brand"].value_counts().index[:30]
    df["product_brand_top"] = np.where(df["product_brand"].isin(top_brands), df["product_brand"], "OTHER")
if "product_category" in df.columns:
    top_cats = df["product_category"].value_counts().index[:30]
    df["product_category_top"] = np.where(df["product_category"].isin(top_cats), df["product_category"], "OTHER")

df.to_parquet(ARTIFACTS_DIR / "clean_retail_data.parquet", index=False)
print("Preprocessing complete. Clean data saved.")
