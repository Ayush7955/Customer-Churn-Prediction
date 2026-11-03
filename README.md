# 📉 Customer Churn Predictor

Predicts whether a customer will leave (churn) or stay, from age, gender, tenure, monthly charges, total charges, contract type and payment method. It compares Logistic Regression, a Random Forest and a Neural Network (TensorFlow/Keras), and the Streamlit website uses the best one (highest ROC-AUC).

## Files

```
churn-predictor/
├── app.py                              # Streamlit website
├── main.py                             # pipeline: load, clean, explore, train, compare
├── generate_data.py                    # creates the synthetic dataset
├── synthetic_customer_churn_100k.csv   # 100,000 customers, 9 columns
├── requirements.txt
└── README.md
```

## Run it

```
pip install -r requirements.txt
streamlit run app.py
```

The first start trains the three models (about a minute), then the result is cached.
To rebuild the dataset: `python generate_data.py`.

## Pipeline

1. Load the CSV and turn `Churn` into 0/1; drop `CustomerID`.
2. Median-impute and scale numbers; most-frequent-impute and one-hot encode text columns.
3. 80/20 stratified split; preprocessing is fitted on the training data only.
4. Train Logistic Regression, Random Forest (200 trees) and a neural network
   (Dense 64 → Dropout 0.3 → Dense 32 → Dropout 0.2 → sigmoid, early stopping).
5. Compare on Accuracy, Precision, Recall, F1 and ROC-AUC; draw ROC curves.

## Website tabs

- **Predict:** enter a customer in the sidebar, see each model's churn chance and a risk verdict. The *risk threshold* slider lets you flag more customers (higher recall) or fewer.
- **Explore data:** churn by contract, tenure and monthly charges.
- **Compare models:** metrics table and ROC curves.

## Note

The data is synthetic, so scores reflect the rule used to generate it, not real customers. With a real dataset, the neural network or forest may do better than Logistic Regression.
