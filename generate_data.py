# Creates synthetic_customer_churn_100k.csv (100,000 customers, 9 columns, ~33% churn)
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
n = 100_000

age = rng.integers(18, 81, n)
gender = rng.choice(["Female", "Male", "Other"], n, p=[0.49, 0.49, 0.02])
tenure = rng.integers(1, 73, n)
monthly = np.round(rng.uniform(10, 150, n), 2)
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
payment = rng.choice(["Bank transfer", "Credit card", "Electronic check", "Mailed check"], n)
total = np.round(tenure * monthly * rng.uniform(0.95, 1.05, n), 2)

# Hidden rule that decides who leaves (plus noise, so it is not perfectly predictable)
z = (-0.6
     + 1.2 * (contract == "Month-to-month") - 0.9 * (contract == "Two year")
     - 0.035 * tenure
     + 0.012 * (monthly - 70)
     + 0.7 * (payment == "Electronic check")
     + 0.01 * (40 - age)
     + rng.normal(0, 0.9, n))
churn = np.where(rng.random(n) < 1 / (1 + np.exp(-z)), "Yes", "No")

df = pd.DataFrame({"CustomerID": [f"C{i:06d}" for i in range(n)], "Age": age, "Gender": gender,
                   "Tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total,
                   "Contract": contract, "PaymentMethod": payment, "Churn": churn})
df.to_csv("synthetic_customer_churn_100k.csv", index=False)
print(df.shape, "churn rate:", (df.Churn == "Yes").mean().round(3))
