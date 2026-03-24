from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from predict_core import predict_from_dataframe

app = Flask(__name__)

# Load model
data = joblib.load("transaction_anomaly_model_full.pkl")
model = data['pipeline']
anomaly_min = data['anomaly_min']
anomaly_max = data['anomaly_max']

df_dataset = pd.read_csv("dataset.csv")  
dropdown_fields = [
    "merchant_category", "transaction_type", "entry_mode",
    "transaction_city", "transaction_state", "transaction_country", "currency_code"
]

dropdown_values = {field: sorted(df_dataset[field].dropna().unique()) for field in dropdown_fields}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        txn_data = {
            "transaction_id": request.form.get("transaction_id", "TXN000001"),
            "card_number": request.form.get("card_number", ""),
            "transaction_datetime": request.form.get("transaction_datetime", ""),
            "transaction_amount": float(request.form.get("transaction_amount", 0)),
            "merchant_id": request.form.get("merchant_id", ""),
            "merchant_category": request.form.get("merchant_category", ""),
            "transaction_type": request.form.get("transaction_type", ""),
            "entry_mode": request.form.get("entry_mode", ""),
            "transaction_city": request.form.get("transaction_city", ""),
            "transaction_state": request.form.get("transaction_state", ""),
            "transaction_country": request.form.get("transaction_country", ""),
            "currency_code": request.form.get("currency_code", ""),
            "is_international": request.form.get("is_international") == "on",
            "card_present": request.form.get("card_present") == "on",
            "cardholder_id": request.form.get("cardholder_id", ""),
            "cardholder_age": int(request.form.get("cardholder_age", 0)),
            "cardholder_gender": request.form.get("cardholder_gender", "")
        }

        df_single = pd.DataFrame([txn_data])
        df_out = predict_from_dataframe(df_single, model, anomaly_min, anomaly_max)

        risk_score = round(df_out['predicted_risk_score'].values[0], 2)
        risk_level = df_out['predicted_risk_level'].values[0]
        color_hex = "#28a745" if risk_level=="Low" else "#ffc107" if risk_level=="Medium" else "#dc3545"

        return render_template(
            "result.html",
            risk_score=risk_score,
            risk_level=risk_level,
            color=color_hex,
            transaction=txn_data
        )
    return render_template("index.html", dropdown_values=dropdown_values)

@app.route("/predict_live", methods=["POST"])
def predict_live():
    txn_data = {k: request.form.get(k) for k in request.form}
    txn_data["transaction_amount"] = float(txn_data.get("transaction_amount", 0))
    txn_data["cardholder_age"] = int(txn_data.get("cardholder_age", 0))
    txn_data["is_international"] = request.form.get("is_international") == "on"
    txn_data["card_present"] = request.form.get("card_present") == "on"

    df_single = pd.DataFrame([txn_data])
    df_out = predict_from_dataframe(df_single, model, anomaly_min, anomaly_max)
    risk_score = round(df_out['predicted_risk_score'].values[0], 2)
    risk_level = df_out['predicted_risk_level'].values[0]
    color_hex = "#28a745" if risk_level=="Low" else "#ffc107" if risk_level=="Medium" else "#dc3545"

    return jsonify({"risk_score": risk_score, "risk_level": risk_level, "color": color_hex})

if __name__ == "__main__":
    app.run(debug=True)
