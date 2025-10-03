import pandas as pd
import lightgbm as lgb
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------
# Load & preprocess dataset
# -----------------------------
df = pd.read_csv("data.csv")

category_mappings = {}
for col in ['Gender','Income','Customer_Segment','Product_Category','Shipping_Method','Payment_Method']:
    df[col] = df[col].astype('category')
    category_mappings[col] = dict(enumerate(df[col].cat.categories))
    category_mappings[col+"_inv"] = {v: k for k, v in category_mappings[col].items()}
    df[col] = df[col].cat.codes

df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.weekday
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour

df = df.sort_values('Date')
df['Rolling_7d'] = df.groupby('Product_Category')['Total_Purchases'].transform(lambda x: x.rolling(7, min_periods=1).mean())
df['Rolling_14d'] = df.groupby('Product_Category')['Total_Purchases'].transform(lambda x: x.rolling(14, min_periods=1).mean())
df['Segment_Product'] = df['Customer_Segment'] * df['Product_Category']

X = df[['Year','Month','Day','Weekday','Hour','Product_Category','Customer_Segment',
        'Shipping_Method','Payment_Method','Amount','Total_Amount','Ratings','Age',
        'Rolling_7d','Rolling_14d','Segment_Product']]
y = df['Total_Purchases']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=10, num_leaves=64, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# -----------------------------
# Prediction function
# -----------------------------
def predict(date_text, hour, product_category, customer_segment, shipping_method, payment_method,
            gender, income, amount, ratings, age):

    try:
        dt = pd.to_datetime(date_text)
    except:
        return "❌ Invalid date format! Use YYYY-MM-DD"

    year, month, day, weekday = dt.year, dt.month, dt.day, dt.weekday()

    prod_cat_code = category_mappings['Product_Category_inv'][product_category]
    cust_seg_code = category_mappings['Customer_Segment_inv'][customer_segment]
    shipping_code = category_mappings['Shipping_Method_inv'][shipping_method]
    payment_code = category_mappings['Payment_Method_inv'][payment_method]
    gender_code = category_mappings['Gender_inv'][gender]
    income_code = category_mappings['Income_inv'][income]

    total_amount = df['Total_Amount'].mean()
    rolling_7d = df['Rolling_7d'].mean()
    rolling_14d = df['Rolling_14d'].mean()
    segment_product = cust_seg_code * prod_cat_code

    row = pd.DataFrame([[year, month, day, weekday, hour, prod_cat_code, cust_seg_code,
                         shipping_code, payment_code, amount, total_amount, ratings, age,
                         rolling_7d, rolling_14d, segment_product]], columns=X.columns)

    pred = model.predict(row)[0]
    return f"✅ Predicted Total Purchases: {round(pred,2)}"

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🛒 Sales Prediction App")
st.markdown(f"**Model Performance:** R² = {round(r2,3)}, MAE = {round(mae,3)}")

with st.form("prediction_form"):
    date_input = st.text_input("Date (YYYY-MM-DD)", "2025-10-03")
    hour_input = st.slider("Hour", 0, 23, 12)

    product_input = st.selectbox("Product Category", list(category_mappings['Product_Category'].values()))
    segment_input = st.selectbox("Customer Segment", list(category_mappings['Customer_Segment'].values()))

    shipping_input = st.selectbox("Shipping Method", list(category_mappings['Shipping_Method'].values()))
    payment_input = st.selectbox("Payment Method", list(category_mappings['Payment_Method'].values()))

    gender_input = st.selectbox("Gender", list(category_mappings['Gender'].values()))
    income_input = st.selectbox("Income Level", list(category_mappings['Income'].values()))

    amount_input = st.number_input("Amount", value=100)
    ratings_input = st.slider("Ratings", 1, 5, 4)
    age_input = st.slider("Age", 10, 100, 30)

    submitted = st.form_submit_button("Predict")
    if submitted:
        result = predict(date_input, hour_input, product_input, segment_input,
                         shipping_input, payment_input, gender_input, income_input,
                         amount_input, ratings_input, age_input)
        st.success(result)
