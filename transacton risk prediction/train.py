import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")

df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'])
df['txn_hour'] = df['transaction_datetime'].dt.hour
df['txn_day'] = df['transaction_datetime'].dt.day
df['txn_month'] = df['transaction_datetime'].dt.month
df['is_weekend'] = (df['transaction_datetime'].dt.weekday >= 5).astype(int)

drop_cols = [
    'transaction_id',
    'card_number',
    'merchant_id',
    'transaction_datetime',
    'is_fraud'   
]
df_model = df.drop(columns=drop_cols)

numeric_features = [
    'transaction_amount',
    'cardholder_age',
    'txn_hour',
    'txn_day',
    'txn_month',
    'is_weekend',
    'is_international'
]

categorical_features = [
    'merchant_category',
    'transaction_type',
    'entry_mode',
    'transaction_city',
    'transaction_state',
    'transaction_country',
    'currency_code',
    'cardholder_gender',
    'cardholder_id'
]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
isolation_forest = IsolationForest(
    n_estimators=200,
    contamination=0.03,
    random_state=42
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', isolation_forest)
])

model_pipeline.fit(df_model)

X_transformed = model_pipeline.named_steps['preprocessor'].transform(df_model)
anomaly_scores = model_pipeline.named_steps['model'].decision_function(X_transformed)

anomaly_min = anomaly_scores.min()
anomaly_max = anomaly_scores.max()

joblib.dump({
    'pipeline': model_pipeline,
    'anomaly_min': anomaly_min,
    'anomaly_max': anomaly_max
}, "transaction_anomaly_model_full.pkl")

print(" Model saved as transaction_anomaly_model_full.pkl")

# =========================
# 11. GENERATE RISK SCORES
# =========================
df['risk_score'] = ((anomaly_max - anomaly_scores) / (anomaly_max - anomaly_min)) * 100
df['risk_level'] = pd.cut(df['risk_score'], bins=[0, 30, 70, 100], labels=['Low', 'Medium', 'High'])

# =========================
# 12. MODEL EVALUATION
# =========================
if 'is_fraud' in df.columns:
    y_true = df['is_fraud'].astype(int)
    y_score = df['risk_score']

    roc_auc = roc_auc_score(y_true, y_score)
    print(f"📊 ROC-AUC Score: {roc_auc:.4f}")

    def precision_at_k(y_true, y_score, k=0.05):
        threshold = np.percentile(y_score, 100 - k * 100)
        preds = y_score >= threshold
        return y_true[preds].sum() / preds.sum()

    p_at_5 = precision_at_k(y_true, y_score, k=0.05)
    print(f"📊 Precision@5%: {p_at_5:.4f}")

# =========================
# 13. SAVE OUTPUT
# =========================
df.to_csv("transaction_risk_output.csv", index=False)
print("✅ Risk scoring output saved as transaction_risk_output.csv")

# =========================
# 14. OPTIONAL: RISK DISTRIBUTION PLOT
# =========================
plt.hist(df['risk_score'], bins=50)
plt.title("Transaction Risk Score Distribution")
plt.xlabel("Risk Score")
plt.ylabel("Number of Transactions")
plt.show()
