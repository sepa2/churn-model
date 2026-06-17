import streamlit as st
import joblib
import numpy as np

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Churn Prediction")

st.write("Prediksi Customer Churn menggunakan Random Forest")

gender = st.selectbox("Gender", ["Female", "Male"])
gender = 0 if gender == "Female" else 1

senior = st.selectbox("Senior Citizen", ["No", "Yes"])
senior = 0 if senior == "No" else 1

partner = st.selectbox("Partner", ["No", "Yes"])
partner = 0 if partner == "No" else 1

dependents = st.selectbox("Dependents", ["No", "Yes"])
dependents = 0 if dependents == "No" else 1

tenure = st.number_input("Tenure", min_value=0)

phone = st.selectbox("Phone Service", ["No", "Yes"])
phone = 0 if phone == "No" else 1

multiple = st.selectbox(
    "Multiple Lines",
    ["No", "No phone service", "Yes"]
)
multiple = {
    "No": 0,
    "No phone service": 1,
    "Yes": 2
}[multiple]

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)
internet = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
}[internet]

online_security = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)
online_security = {
    "No": 0,
    "Yes": 1,
    "No internet service": 2
}[online_security]

online_backup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)
online_backup = {
    "No": 0,
    "Yes": 1,
    "No internet service": 2
}[online_backup]

device_protection = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)
device_protection = {
    "No": 0,
    "Yes": 1,
    "No internet service": 2
}[device_protection]

tech_support = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)
tech_support = {
    "No": 0,
    "Yes": 1,
    "No internet service": 2
}[tech_support]

streaming_tv = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)
streaming_tv = {
    "No": 0,
    "Yes": 1,
    "No internet service": 2
}[streaming_tv]

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)
streaming_movies = {
    "No": 0,
    "Yes": 1,
    "No internet service": 2
}[streaming_movies]

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)
contract = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}[contract]

paperless = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)
paperless = 0 if paperless == "No" else 1

payment = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)
payment = {
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 1,
    "Electronic check": 2,
    "Mailed check": 3
}[payment]

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
