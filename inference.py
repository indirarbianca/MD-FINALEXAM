"""
inference.py
=============
Script inferencing untuk model Credit Score Classification.

Modul ini bertugas:
- Memuat model terbaik beserta preprocessor-nya dari file pickle (best_model.pkl)
- Menerima input data nasabah baru (berupa dict / DataFrame)
- Melakukan preprocessing yang konsisten dengan saat training
- Mengembalikan hasil prediksi (label kelas + confidence/probabilitas)

Modul ini digunakan langsung oleh app.py (Streamlit) untuk melakukan
inferencing secara real-time, dan juga bisa dipanggil secara mandiri
lewat command line untuk keperluan testing/batch prediction.
"""

import pickle
import pandas as pd
import numpy as np

from preprocessing import CreditDataPreprocessor  # wajib di-import agar unpickling berhasil


class CreditScoreInference:
    """
    Class untuk melakukan inferencing model credit score yang sudah dilatih.
    """

    def __init__(self, model_path: str = "best_model.pkl"):
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)

        self.model = artifact["model"]
        self.preprocessor: CreditDataPreprocessor = artifact["preprocessor"]
        self.model_name = artifact["model_name"]
        self.class_names = artifact["class_names"]
        self.feature_columns = artifact["feature_columns"]

    def predict(self, input_data: dict) -> dict:
        """
        Melakukan prediksi untuk satu data nasabah.

        Parameters
        ----------
        input_data : dict
            Dictionary berisi fitur-fitur nasabah, contoh:
            {
                "Age": 32, "Annual_Income": 56125.5, "Occupation": "Doctor", ...
            }

        Returns
        -------
        dict
            {
                "predicted_class": "Standard",
                "probabilities": {"Good": 0.1, "Poor": 0.2, "Standard": 0.7},
                "model_used": "Random_Forest"
            }
        """
        df_input = pd.DataFrame([input_data])

        # Pastikan kolom-kolom yang dibutuhkan preprocessing tersedia,
        # isi NaN untuk kolom yang tidak diberikan (akan diimputasi otomatis)
        required_raw_cols = [
            "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
            "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Type_of_Loan",
            "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
            "Num_Credit_Inquiries", "Credit_Mix", "Outstanding_Debt",
            "Credit_Utilization_Ratio", "Credit_History_Age", "Payment_of_Min_Amount",
            "Total_EMI_per_month", "Amount_invested_monthly", "Payment_Behaviour",
            "Monthly_Balance", "Occupation",
        ]
        for col in required_raw_cols:
            if col not in df_input.columns:
                df_input[col] = np.nan

        X_processed, _ = self.preprocessor.transform(df_input)

        pred_encoded = self.model.predict(X_processed)[0]
        predicted_class = self.preprocessor.label_encoder.inverse_transform([pred_encoded])[0]

        result = {
            "predicted_class": predicted_class,
            "model_used": self.model_name,
        }

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_processed)[0]
            result["probabilities"] = {
                cls: float(round(p, 4))
                for cls, p in zip(self.preprocessor.label_encoder.classes_, proba)
            }

        return result

    def predict_batch(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Prediksi untuk banyak baris sekaligus (misalnya dari file CSV upload)."""
        X_processed, _ = self.preprocessor.transform(df_raw)
        preds_encoded = self.model.predict(X_processed)
        preds_label = self.preprocessor.label_encoder.inverse_transform(preds_encoded)

        result_df = df_raw.copy()
        result_df["Predicted_Credit_Score"] = preds_label
        return result_df


# =====================================================================
# CONTOH PEMAKAIAN / TEST CASE MANDIRI (CLI)
# =====================================================================
if __name__ == "__main__":
    infer = CreditScoreInference("best_model.pkl")
    print(f"Model yang digunakan: {infer.model_name}")
    print(f"Kelas target: {infer.class_names}\n")

    # Tiga contoh test case yang merepresentasikan masing-masing kelas
    test_cases = {
        "Contoh nasabah profil BAIK (ekspektasi: Good)": {
            "Age": 35, "Annual_Income": 120000, "Monthly_Inhand_Salary": 9500,
            "Num_Bank_Accounts": 3, "Num_Credit_Card": 3, "Interest_Rate": 5,
            "Num_of_Loan": 1, "Type_of_Loan": "Home Equity Loan",
            "Delay_from_due_date": 2, "Num_of_Delayed_Payment": 1,
            "Changed_Credit_Limit": 5.0, "Num_Credit_Inquiries": 1,
            "Credit_Mix": "Good", "Outstanding_Debt": 200.0,
            "Credit_Utilization_Ratio": 20.0, "Credit_History_Age": "20 Years and 5 Months",
            "Payment_of_Min_Amount": "No", "Total_EMI_per_month": 50.0,
            "Amount_invested_monthly": 500.0,
            "Payment_Behaviour": "Low_spent_Large_value_payments",
            "Monthly_Balance": 1200.0, "Occupation": "Doctor",
        },
        "Contoh nasabah profil SEDANG (ekspektasi: Standard)": {
            "Age": 30, "Annual_Income": 45000, "Monthly_Inhand_Salary": 3500,
            "Num_Bank_Accounts": 5, "Num_Credit_Card": 5, "Interest_Rate": 15,
            "Num_of_Loan": 3, "Type_of_Loan": "Personal Loan, and Auto Loan",
            "Delay_from_due_date": 15, "Num_of_Delayed_Payment": 10,
            "Changed_Credit_Limit": 8.0, "Num_Credit_Inquiries": 4,
            "Credit_Mix": "Standard", "Outstanding_Debt": 1200.0,
            "Credit_Utilization_Ratio": 32.0, "Credit_History_Age": "12 Years and 3 Months",
            "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 150.0,
            "Amount_invested_monthly": 100.0,
            "Payment_Behaviour": "High_spent_Medium_value_payments",
            "Monthly_Balance": 400.0, "Occupation": "Engineer",
        },
        "Contoh nasabah profil BURUK (ekspektasi: Poor)": {
            "Age": 24, "Annual_Income": 9000, "Monthly_Inhand_Salary": 650,
            "Num_Bank_Accounts": 10, "Num_Credit_Card": 10, "Interest_Rate": 34,
            "Num_of_Loan": 9, "Type_of_Loan": "Payday Loan, Mortgage Loan, Personal Loan, and Debt Consolidation Loan",
            "Delay_from_due_date": 55, "Num_of_Delayed_Payment": 26,
            "Changed_Credit_Limit": -5.5, "Num_Credit_Inquiries": 15,
            "Credit_Mix": "Bad", "Outstanding_Debt": 4500.0,
            "Credit_Utilization_Ratio": 49.5, "Credit_History_Age": "1 Years and 2 Months",
            "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 480.0,
            "Amount_invested_monthly": 5.0,
            "Payment_Behaviour": "Low_spent_Small_value_payments",
            "Monthly_Balance": 15.0, "Occupation": "Mechanic",
        },
    }

    for desc, data in test_cases.items():
        result = infer.predict(data)
        print(f"--- {desc} ---")
        print(f"  Prediksi   : {result['predicted_class']}")
        print(f"  Probabilitas: {result.get('probabilities')}")
        print()
