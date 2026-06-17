import streamlit as st
import joblib
import numpy as np

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Churn Prediction")

st.write("Prediksi Customer Churn menggunakan Random Forest")

gender = st.selectbox("Gender", [0,1])
senior = st.selectbox("Senior Citizen", [0,1])
partner = st.selectbox("Partner", [0,1])
dependents = st.selectbox("Dependents", [0,1])

tenure = st.number_input("Tenure", min_value=0)

phone = st.selectbox("Phone Service", [0,1])
multiple = st.selectbox("Multiple Lines", [0,1,2])

internet = st.selectbox("Internet Service", [0,1,2])

online_security = st.selectbox("Online Security", [0,1,2])
online_backup = st.selectbox("Online Backup", [0,1,2])
device_protection = st.selectbox("Device Protection", [0,1,2])
tech_support = st.selectbox("Tech Support", [0,1,2])

streaming_tv = st.selectbox("Streaming TV", [0,1,2])
streaming_movies = st.selectbox("Streaming Movies", [0,1,2])

contract = st.selectbox("Contract", [0,1,2])

paperless = st.selectbox("Paperless Billing", [0,1])

payment = st.selectbox("Payment Method", [0,1,2,3])

monthly = st.number_input("Monthly Charges", min_value=0.0)
total = st.number_input("Total Charges", min_value=0.0)

avg_monthly_spend = total / (tenure + 1)

if st.button("Prediksi"):

    data = np.array([[
        gender,
        senior,
        partner,
        dependents,
        tenure,
        phone,
        multiple,
        internet,
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies,
        contract,
        paperless,
        payment,
        monthly,
        total,
        avg_monthly_spend
    ]])

    data[:, [4,17,18,19]] = scaler.transform(
        data[:, [4,17,18,19]]
    )

    pred = model.predict(data)

    if pred[0] == 1:
        st.error("Pelanggan Berpotensi Churn")
    else:
        st.success("Pelanggan Diprediksi Bertahan")


st.write("Akurasi Model Random Forest: 79.77%")
