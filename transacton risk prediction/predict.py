import pandas as pd
import joblib
from predict_core import predict_from_dataframe


data = joblib.load("transaction_anomaly_model_full.pkl")
model = data['pipeline']
anomaly_min = data['anomaly_min']
anomaly_max = data['anomaly_max']

single_transaction = {
    "transaction_id": "TXN000004",
    "card_number": "5105110000000000",   
    "transaction_datetime": "07-04-2022 18:11",
    "transaction_amount": 5921,
    "merchant_id": "MRH00045",
    "merchant_category": "travel",
    "transaction_type": "purchase",
    "entry_mode": "online",
    "transaction_city": "London",
    "transaction_state": "",
    "transaction_country": "GBR",
    "currency_code": "GBP",
    "is_international": True,
    "card_present": True,  
    "cardholder_id": "CH0004",
    "cardholder_age": 54,
    "cardholder_gender": "male"
}


df_single = pd.DataFrame([single_transaction])

df_out = predict_from_dataframe(df_single, model, anomaly_min, anomaly_max)

print(" Single Input Prediction")
print(df_out[['predicted_risk_score', 'predicted_risk_level']])
