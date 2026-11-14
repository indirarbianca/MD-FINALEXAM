"""
app.py
=======
Aplikasi web Streamlit untuk deployment model Credit Score Classification.

Menyediakan dua mode input:
1. Form interaktif — user mengisi data nasabah secara manual lewat UI.
2. Upload CSV — untuk prediksi batch banyak nasabah sekaligus.

Jalankan dengan:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd

from inference import CreditScoreInference

st.set_page_config(
    page_title="Credit Score Prediction",
    page_icon="💳",
    layout="wide",
)

# ---------------------------------------------------------------------
# Load model (cache agar tidak reload setiap interaksi)
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    return CreditScoreInference("best_model.pkl")

infer = load_model()

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("💳 Credit Score Prediction System")
st.markdown("""
Aplikasi ini memprediksi **kategori credit score** nasabah (`Good`, `Standard`, atau `Poor`)
berdasarkan data finansial dan riwayat pembayaran mereka, menggunakan model
**Random Forest** yang telah dilatih pada data historis nasabah institusi keuangan.
""")
st.info(f"Model yang digunakan: **{infer.model_name}** | Kelas target: {', '.join(infer.class_names)}")

tab1, tab2 = st.tabs(["🧍 Prediksi Individu", "📂 Prediksi Batch (Upload CSV)"])

# ---------------------------------------------------------------------
# TAB 1: Prediksi Individu (Form)
# ---------------------------------------------------------------------
with tab1:
    st.subheader("Masukkan Data Nasabah")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Data Demografi & Pekerjaan**")
        age = st.number_input("Usia (Age)", min_value=18, max_value=100, value=30)
        occupation = st.selectbox("Pekerjaan (Occupation)", [
            "Accountant", "Architect", "Developer", "Doctor", "Engineer",
            "Entrepreneur", "Journalist", "Lawyer", "Manager", "Mechanic",
            "Media_Manager", "Musician", "Scientist", "Teacher", "Writer",
        ])
        annual_income = st.number_input("Pendapatan Tahunan (Annual Income)", min_value=0.0, value=50000.0, step=1000.0)
        monthly_inhand_salary = st.number_input("Gaji Bulanan Bersih (Monthly Inhand Salary)", min_value=0.0, value=4000.0, step=100.0)

    with col2:
        st.markdown("**Profil Kredit**")
        num_bank_accounts = st.number_input("Jumlah Rekening Bank", min_value=0, max_value=20, value=4)
        num_credit_card = st.number_input("Jumlah Kartu Kredit", min_value=0, max_value=20, value=4)
        interest_rate = st.number_input("Suku Bunga (%)", min_value=0, max_value=50, value=15)
        num_of_loan = st.number_input("Jumlah Pinjaman", min_value=0, max_value=20, value=2)
        type_of_loan = st.text_input("Jenis Pinjaman (pisahkan dengan koma)", value="Personal Loan, and Auto Loan")
        credit_mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"])

    with col3:
        st.markdown("**Riwayat Pembayaran**")
        delay_from_due_date = st.number_input("Rata-rata Hari Terlambat Bayar", min_value=0, max_value=120, value=10)
        num_of_delayed_payment = st.number_input("Jumlah Pembayaran Terlambat", min_value=0, max_value=60, value=5)
        changed_credit_limit = st.number_input("Perubahan Limit Kredit", value=5.0, step=0.5)
        num_credit_inquiries = st.number_input("Jumlah Credit Inquiries", min_value=0, max_value=50, value=3)
        payment_of_min_amount = st.selectbox("Bayar Minimum Saja?", ["Yes", "No", "NM"])
        credit_history_years = st.number_input("Lama Riwayat Kredit (Tahun)", min_value=0, max_value=50, value=10)
        credit_history_months = st.number_input("Lama Riwayat Kredit (Bulan Tambahan)", min_value=0, max_value=11, value=0)

    st.markdown("**Data Finansial Lainnya**")
    col4, col5, col6 = st.columns(3)
    with col4:
        outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, value=500.0, step=50.0)
        credit_utilization_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=30.0)
    with col5:
        total_emi_per_month = st.number_input("Total EMI per Bulan", min_value=0.0, value=100.0, step=10.0)
        amount_invested_monthly = st.number_input("Jumlah Investasi Bulanan", min_value=0.0, value=150.0, step=10.0)
    with col6:
        monthly_balance = st.number_input("Saldo Bulanan (Monthly Balance)", min_value=0.0, value=400.0, step=10.0)
        payment_behaviour = st.selectbox("Perilaku Pembayaran", [
            "High_spent_Large_value_payments", "High_spent_Medium_value_payments",
            "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
            "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments",
        ])

    if st.button("🔍 Prediksi Credit Score", type="primary", use_container_width=True):
        input_data = {
            "Age": age, "Annual_Income": annual_income,
            "Monthly_Inhand_Salary": monthly_inhand_salary,
            "Num_Bank_Accounts": num_bank_accounts, "Num_Credit_Card": num_credit_card,
            "Interest_Rate": interest_rate, "Num_of_Loan": num_of_loan,
            "Type_of_Loan": type_of_loan,
            "Delay_from_due_date": delay_from_due_date,
            "Num_of_Delayed_Payment": num_of_delayed_payment,
            "Changed_Credit_Limit": changed_credit_limit,
            "Num_Credit_Inquiries": num_credit_inquiries,
            "Credit_Mix": credit_mix, "Outstanding_Debt": outstanding_debt,
            "Credit_Utilization_Ratio": credit_utilization_ratio,
            "Credit_History_Age": f"{credit_history_years} Years and {credit_history_months} Months",
            "Payment_of_Min_Amount": payment_of_min_amount,
            "Total_EMI_per_month": total_emi_per_month,
            "Amount_invested_monthly": amount_invested_monthly,
            "Payment_Behaviour": payment_behaviour,
            "Monthly_Balance": monthly_balance, "Occupation": occupation,
        }

        result = infer.predict(input_data)
        pred_class = result["predicted_class"]

        color_map = {"Good": "green", "Standard": "orange", "Poor": "red"}
        st.markdown("---")
        st.markdown(f"### Hasil Prediksi: :{color_map.get(pred_class,'blue')}[{pred_class}]")

        if "probabilities" in result:
            proba_df = pd.DataFrame(
                list(result["probabilities"].items()), columns=["Kelas", "Probabilitas"]
            ).sort_values("Probabilitas", ascending=False)
            st.bar_chart(proba_df.set_index("Kelas"))
            st.dataframe(proba_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# TAB 2: Prediksi Batch (Upload CSV)
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Upload File CSV untuk Prediksi Batch")
    st.markdown("""
    Upload file CSV dengan struktur kolom yang sama seperti dataset training
    (tidak perlu menyertakan kolom `Credit_Score` jika belum diketahui).
    """)

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("Preview data yang diupload:")
        st.dataframe(df_uploaded.head(), use_container_width=True)

        if st.button("🔍 Prediksi Semua Baris", type="primary"):
            with st.spinner("Memproses prediksi..."):
                result_df = infer.predict_batch(df_uploaded)
            st.success(f"Prediksi selesai untuk {len(result_df)} baris data.")
            st.dataframe(result_df, use_container_width=True)

            csv_result = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Hasil Prediksi (CSV)",
                data=csv_result,
                file_name="hasil_prediksi_credit_score.csv",
                mime="text/csv",
            )

st.markdown("---")
st.caption("Model Deployment UAS — Credit Score Classification | Random Forest Classifier")
