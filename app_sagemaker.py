"""
app_sagemaker.py
=================
Aplikasi Streamlit yang melakukan inferencing dengan memanggil
**AWS SageMaker Endpoint** (bukan memuat model pickle secara lokal).

Aplikasi ini untuk Bagian 2b (deployment cloud), di mana model sudah
di-deploy sebagai real-time endpoint di SageMaker, dan Streamlit hanya
bertindak sebagai antarmuka (front-end) yang mengirim request ke endpoint.

run:
    streamlit run app_sagemaker.py
"""

import json
import streamlit as st
import pandas as pd
import boto3

st.set_page_config(page_title="Credit Score Prediction (SageMaker)", page_icon="💳", layout="wide")

# ---------------------------------------------------------------------
# Konfigurasi endpoint & kredensial
# ---------------------------------------------------------------------
def get_config():
    """Ambil konfigurasi dari Streamlit Secrets jika ada, fallback ke default."""
    cfg = {
        "region": "us-east-1",
        "endpoint_name": "credit-score-endpoint",
        "aws_access_key_id": None,
        "aws_secret_access_key": None,
        "aws_session_token": None,
    }
    try:
        if "AWS_REGION" in st.secrets:
            cfg["region"] = st.secrets["AWS_REGION"]
        if "SAGEMAKER_ENDPOINT_NAME" in st.secrets:
            cfg["endpoint_name"] = st.secrets["SAGEMAKER_ENDPOINT_NAME"]
        if "AWS_ACCESS_KEY_ID" in st.secrets:
            cfg["aws_access_key_id"] = st.secrets["AWS_ACCESS_KEY_ID"]
            cfg["aws_secret_access_key"] = st.secrets["AWS_SECRET_ACCESS_KEY"]
            cfg["aws_session_token"] = st.secrets.get("AWS_SESSION_TOKEN")
    except Exception:
        # st.secrets tidak tersedia (mis. dijalankan di dalam lab) -> pakai default + role otomatis
        pass
    return cfg


@st.cache_resource
def get_runtime_client():
    cfg = get_config()
    kwargs = {"region_name": cfg["region"]}
    # Jika credentials eksplisit diberikan (mode di luar lab), pakai itu.
    if cfg["aws_access_key_id"]:
        kwargs["aws_access_key_id"] = cfg["aws_access_key_id"]
        kwargs["aws_secret_access_key"] = cfg["aws_secret_access_key"]
        if cfg["aws_session_token"]:
            kwargs["aws_session_token"] = cfg["aws_session_token"]
    client = boto3.client("sagemaker-runtime", **kwargs)
    return client, cfg["endpoint_name"]


def call_endpoint(payload: dict):
    client, endpoint_name = get_runtime_client()
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    result = json.loads(response["Body"].read().decode())
    return result[0] if isinstance(result, list) else result


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("💳 Credit Score Prediction System (via AWS SageMaker)")
st.markdown("""
Aplikasi ini memprediksi kategori credit score nasabah (`Good`, `Standard`, `Poor`) dengan
mengirim request ke **AWS SageMaker Endpoint** yang menjalankan model **Random Forest**.
""")

_, cfg = get_runtime_client()
st.info(f"Endpoint: **{cfg['endpoint_name']}** | Region: **{cfg['region']}**")

st.subheader("Masukkan Data Nasabah")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demografi & Pekerjaan**")
    age = st.number_input("Usia (Age)", 18, 100, 30)
    occupation = st.selectbox("Pekerjaan", [
        "Accountant", "Architect", "Developer", "Doctor", "Engineer", "Entrepreneur",
        "Journalist", "Lawyer", "Manager", "Mechanic", "Media_Manager", "Musician",
        "Scientist", "Teacher", "Writer",
    ])
    annual_income = st.number_input("Pendapatan Tahunan", 0.0, value=50000.0, step=1000.0)
    monthly_inhand_salary = st.number_input("Gaji Bulanan Bersih", 0.0, value=4000.0, step=100.0)

with col2:
    st.markdown("**Profil Kredit**")
    num_bank_accounts = st.number_input("Jumlah Rekening Bank", 0, 20, 4)
    num_credit_card = st.number_input("Jumlah Kartu Kredit", 0, 20, 4)
    interest_rate = st.number_input("Suku Bunga (%)", 0, 50, 15)
    num_of_loan = st.number_input("Jumlah Pinjaman", 0, 20, 2)
    type_of_loan = st.text_input("Jenis Pinjaman (pisah koma)", "Personal Loan, and Auto Loan")
    credit_mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"])

with col3:
    st.markdown("**Riwayat Pembayaran**")
    delay_from_due_date = st.number_input("Rata-rata Hari Terlambat", 0, 120, 10)
    num_of_delayed_payment = st.number_input("Jumlah Pembayaran Terlambat", 0, 60, 5)
    changed_credit_limit = st.number_input("Perubahan Limit Kredit", value=5.0, step=0.5)
    num_credit_inquiries = st.number_input("Jumlah Credit Inquiries", 0, 50, 3)
    payment_of_min_amount = st.selectbox("Bayar Minimum Saja?", ["Yes", "No", "NM"])
    credit_history_years = st.number_input("Riwayat Kredit (Tahun)", 0, 50, 10)
    credit_history_months = st.number_input("Riwayat Kredit (Bulan Tambahan)", 0, 11, 0)

st.markdown("**Data Finansial Lainnya**")
c4, c5, c6 = st.columns(3)
with c4:
    outstanding_debt = st.number_input("Outstanding Debt", 0.0, value=500.0, step=50.0)
    credit_utilization_ratio = st.number_input("Credit Utilization Ratio (%)", 0.0, 100.0, 30.0)
with c5:
    total_emi_per_month = st.number_input("Total EMI per Bulan", 0.0, value=100.0, step=10.0)
    amount_invested_monthly = st.number_input("Investasi Bulanan", 0.0, value=150.0, step=10.0)
with c6:
    monthly_balance = st.number_input("Saldo Bulanan", 0.0, value=400.0, step=10.0)
    payment_behaviour = st.selectbox("Perilaku Pembayaran", [
        "High_spent_Large_value_payments", "High_spent_Medium_value_payments",
        "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
        "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments",
    ])

if st.button("🔍 Prediksi Credit Score (panggil SageMaker)", type="primary", use_container_width=True):
    payload = {
        "Age": age, "Annual_Income": annual_income, "Monthly_Inhand_Salary": monthly_inhand_salary,
        "Num_Bank_Accounts": num_bank_accounts, "Num_Credit_Card": num_credit_card,
        "Interest_Rate": interest_rate, "Num_of_Loan": num_of_loan, "Type_of_Loan": type_of_loan,
        "Delay_from_due_date": delay_from_due_date, "Num_of_Delayed_Payment": num_of_delayed_payment,
        "Changed_Credit_Limit": changed_credit_limit, "Num_Credit_Inquiries": num_credit_inquiries,
        "Credit_Mix": credit_mix, "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": credit_utilization_ratio,
        "Credit_History_Age": f"{credit_history_years} Years and {credit_history_months} Months",
        "Payment_of_Min_Amount": payment_of_min_amount, "Total_EMI_per_month": total_emi_per_month,
        "Amount_invested_monthly": amount_invested_monthly, "Payment_Behaviour": payment_behaviour,
        "Monthly_Balance": monthly_balance, "Occupation": occupation,
    }
    try:
        with st.spinner("Mengirim request ke SageMaker Endpoint..."):
            result = call_endpoint(payload)
        pred_class = result["predicted_class"]
        color = {"Good": "green", "Standard": "orange", "Poor": "red"}.get(pred_class, "blue")
        st.markdown("---")
        st.markdown(f"### Hasil Prediksi: :{color}[{pred_class}]")
        if "probabilities" in result:
            proba_df = pd.DataFrame(
                list(result["probabilities"].items()), columns=["Kelas", "Probabilitas"]
            ).sort_values("Probabilitas", ascending=False)
            st.bar_chart(proba_df.set_index("Kelas"))
            st.dataframe(proba_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Gagal memanggil endpoint: {e}")
        st.caption("Cek: (1) endpoint masih live, (2) credentials/region benar, "
                   "(3) untuk Learner Lab pastikan session token belum expired.")

st.markdown("---")
st.caption("Model Deployment UAS — Credit Score Classification | AWS SageMaker Endpoint")
