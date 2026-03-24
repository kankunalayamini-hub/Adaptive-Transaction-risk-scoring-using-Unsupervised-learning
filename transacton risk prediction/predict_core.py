import pandas as pd
import numpy as np

FEATURE_COLUMNS = [
    "transaction_amount",
    "cardholder_age",
    "is_international",
    "merchant_category",
    "transaction_type",
    "entry_mode",
    "transaction_city",
    "transaction_state",
    "transaction_country",
    "currency_code",
    "cardholder_gender",
    "cardholder_id",
    "txn_hour",
    "txn_day",
    "txn_month",
    "is_weekend"
]

BOOL_COLUMNS = ['is_international', 'card_present']

def predict_from_dataframe(df, model, anomaly_min, anomaly_max):

    df = df.copy()

    df['transaction_datetime'] = pd.to_datetime(
        df['transaction_datetime'],
        dayfirst=True
    )

    df['txn_hour'] = df['transaction_datetime'].dt.hour
    df['txn_day'] = df['transaction_datetime'].dt.day
    df['txn_month'] = df['transaction_datetime'].dt.month
    df['is_weekend'] = (df['transaction_datetime'].dt.weekday >= 5).astype(int)

    for col in BOOL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(int)

    DROP_COLS = [
        'transaction_id',
        'card_number',
        'merchant_id',
        'transaction_datetime'
    ]
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    df = df[FEATURE_COLUMNS]

    X = model.named_steps['preprocessor'].transform(df)
    raw_score = model.named_steps['model'].decision_function(X)

    risk_score = ((anomaly_max - raw_score) /
                  (anomaly_max - anomaly_min)) * 100
    risk_score = np.clip(risk_score, 0, 100)

    df['predicted_risk_score'] = risk_score.round(2)

    df['predicted_risk_level'] = pd.cut(
        df['predicted_risk_score'],
        bins=[-1, 30, 70, 100],
        labels=['Low', 'Medium', 'High']
    )

    return df
