import pandas as pd
import numpy as np

np.random.seed(42)
n = 50000

# Customer demographics
age = np.random.randint(18, 70, n)
income_level = np.random.choice(['Low', 'Medium', 'High'], n, p=[0.3, 0.5, 0.2])
gender = np.random.choice(['Male', 'Female'], n)

# Customer segment correlated with income and age
customer_segment = []
for a, inc in zip(age, income_level):
    if inc == 'High' and a > 30:
        customer_segment.append('Premium')
    elif inc == 'Low' and a < 40:
        customer_segment.append('Regular')
    else:
        customer_segment.append('Occasional')

# Products and spending correlated with segment
product_category = np.random.choice(['Clothing', 'Electronics', 'Sports'], n)
total_purchases = np.random.poisson(lam=[3 if s=='Regular' else 5 if s=='Occasional' else 8 for s in customer_segment])
amount = total_purchases * np.random.uniform(20, 100, n)
total_amount = amount + np.random.uniform(5, 50, n)  # slightly higher than amount
ratings = np.clip(np.random.normal(4, 0.5, n), 1, 5)

# Temporal features
dates = pd.date_range('2023-01-01', periods=n, freq='D')
year = dates.year
month = dates.month
time = pd.to_datetime(np.random.randint(0, 86400, n), unit='s').time

# Other features
shipping_method = np.random.choice(['Same-Day', 'Standard', 'Express'], n)
payment_method = np.random.choice(['Debit Card', 'Credit Card', 'PayPal'], n)

# Build DataFrame
df = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'Income': income_level,
    'Customer_Segment': customer_segment,
    'Product_Category': product_category,
    'Total_Purchases': total_purchases,
    'Amount': amount,
    'Total_Amount': total_amount,
    'Ratings': ratings,
    'Date': dates,
    'Year': year,
    'Month': month,
    'Time': [t.strftime("%H:%M:%S") for t in time],
    'Shipping_Method': shipping_method,
    'Payment_Method': payment_method
})

df.to_csv('data.csv', index=False)
print("Synthetic dataset saved as data.csv")
