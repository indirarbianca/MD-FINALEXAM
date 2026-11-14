"""
Modul Preprocessing untuk Credit Score Classification
Berisi class CreditDataPreprocessor yang menangani:
- Cleaning kolom numerik 'kotor' (trailing underscore, placeholder aneh)
- Outlier capping berbasis domain knowledge
- Feature engineering (Credit_History_Age -> bulan, Type_of_Loan -> jumlah)
- Imputasi missing value
- Encoding kategorikal
- Scaling numerik
"""

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer


class CreditDataPreprocessor:
    """
    Class untuk preprocessing dataset credit score.
    fit() dipanggil hanya pada data training, transform() dipakai untuk
    training maupun inference agar konsisten (hindari data leakage).
    """

    # Kolom yang dibuang karena bersifat identifier / PII, bukan fitur prediktif
    DROP_COLS = ["ID", "Customer_ID", "Name", "SSN", "Month"]

    # Kolom numerik yang tersimpan sebagai string kotor (ada trailing '_', dll)
    DIRTY_NUMERIC_COLS = [
        "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly",
        "Monthly_Balance",
    ]

    # Kolom numerik yang sudah bersih dari awal (tetap numerik)
    CLEAN_NUMERIC_COLS = [
        "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
        "Interest_Rate", "Delay_from_due_date", "Num_Credit_Inquiries",
        "Credit_Utilization_Ratio", "Total_EMI_per_month",
    ]

    # Kolom kategorikal nominal (di One-Hot Encode)
    NOMINAL_COLS = ["Occupation", "Payment_Behaviour"]

    # Kolom kategorikal ordinal (punya urutan tingkatan)
    ORDINAL_COLS = ["Credit_Mix", "Payment_of_Min_Amount"]
    ORDINAL_ORDER = {
        "Credit_Mix": ["Bad", "Standard", "Good"],
        "Payment_of_Min_Amount": ["No", "NM", "Yes"],
    }

    # Batas domain wajar untuk capping outlier (lower, upper)
    DOMAIN_BOUNDS = {
        "Age": (0, 100),
        "Num_Bank_Accounts": (0, 20),
        "Num_Credit_Card": (0, 20),
        "Interest_Rate": (0, 50),
        "Num_of_Loan": (0, 20),
        "Num_of_Delayed_Payment": (0, 60),
        "Num_Credit_Inquiries": (0, 50),
        "Delay_from_due_date": (0, 120),
        "Annual_Income": (0, 300000),
        "Outstanding_Debt": (0, 10000),
        "Amount_invested_monthly": (0, 10000),
        "Monthly_Balance": (0, 10000),
        "Total_EMI_per_month": (0, 5000),
    }

    # Karakter sampah yang menandakan missing value pada kolom kategorikal
    CATEGORICAL_MISSING_TOKENS = {
        "Occupation": "_______",
        "Credit_Mix": "_",
        "Payment_Behaviour": "!@9#%8",
    }

    def __init__(self):
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.ordinal_encoders = {
            col: OrdinalEncoder(categories=[order], handle_unknown="use_encoded_value", unknown_value=-1)
            for col, order in self.ORDINAL_ORDER.items()
        }
        self.cat_imputer_values_ = {}
        self.label_encoder = LabelEncoder()
        self.final_columns_ = None
        self.is_fitted = False

    # ---------- helper static methods ----------
    @staticmethod
    def _clean_numeric_string(series: pd.Series) -> pd.Series:
        """Hapus karakter non-numerik (mis. trailing '_') lalu convert ke float."""
        cleaned = series.astype(str).str.replace("_", "", regex=False)
        cleaned = cleaned.replace("", np.nan)
        return pd.to_numeric(cleaned, errors="coerce")

    @staticmethod
    def _parse_credit_history_age(val):
        """Konversi 'X Years and Y Months' menjadi total bulan (numerik)."""
        if pd.isna(val):
            return np.nan
        match = re.match(r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?", str(val))
        if match:
            years, months = int(match.group(1)), int(match.group(2))
            return years * 12 + months
        return np.nan

    @staticmethod
    def _count_loan_types(val):
        """Hitung jumlah jenis pinjaman dari kolom Type_of_Loan."""
        if pd.isna(val) or str(val).strip().lower() == "not specified":
            return 0
        return len([x.strip() for x in str(val).split(",")])

    def _engineer_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tahap cleaning + feature engineering yang sama untuk fit & transform."""
        df = df.copy()

        # Drop kolom identifier
        df = df.drop(columns=[c for c in self.DROP_COLS if c in df.columns], errors="ignore")

        # Bersihkan kolom numerik 'kotor'
        for col in self.DIRTY_NUMERIC_COLS:
            if col in df.columns:
                df[col] = self._clean_numeric_string(df[col])

        # Feature engineering
        if "Credit_History_Age" in df.columns:
            df["Credit_History_Age_Months"] = df["Credit_History_Age"].apply(self._parse_credit_history_age)
            df = df.drop(columns=["Credit_History_Age"])

        if "Type_of_Loan" in df.columns:
            df["Num_Loan_Types"] = df["Type_of_Loan"].apply(self._count_loan_types)
            df = df.drop(columns=["Type_of_Loan"])

        # Normalisasi token missing pada kategorikal
        for col, token in self.CATEGORICAL_MISSING_TOKENS.items():
            if col in df.columns:
                df[col] = df[col].replace(token, np.nan)

        # Outlier capping berbasis domain bound
        for col, (lo, hi) in self.DOMAIN_BOUNDS.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lo, upper=hi)

        return df

    def fit(self, df: pd.DataFrame, target_col: str = "Credit_Score"):
        df = df.copy()
        y_raw = df[target_col] if target_col in df.columns else None
        X = df.drop(columns=[target_col]) if target_col in df.columns else df

        X = self._engineer_and_clean(X)

        numeric_cols = self.CLEAN_NUMERIC_COLS + self.DIRTY_NUMERIC_COLS + ["Credit_History_Age_Months", "Num_Loan_Types"]
        numeric_cols = [c for c in numeric_cols if c in X.columns]

        # Fit imputer & scaler numerik
        X_num = self.numeric_imputer.fit_transform(X[numeric_cols])
        X_num = self.scaler.fit_transform(X_num)
        X_num = pd.DataFrame(X_num, columns=numeric_cols, index=X.index)

        # Imputasi kategorikal pakai mode (simpan untuk transform)
        for col in self.NOMINAL_COLS + self.ORDINAL_COLS:
            if col in X.columns:
                mode_val = X[col].mode(dropna=True)
                self.cat_imputer_values_[col] = mode_val.iloc[0] if len(mode_val) else "Unknown"
                X[col] = X[col].fillna(self.cat_imputer_values_[col])

        # One-hot encode nominal
        nominal_present = [c for c in self.NOMINAL_COLS if c in X.columns]
        X_nominal = self.ohe.fit_transform(X[nominal_present])
        nominal_feature_names = self.ohe.get_feature_names_out(nominal_present)
        X_nominal = pd.DataFrame(X_nominal, columns=nominal_feature_names, index=X.index)

        # Ordinal encode
        X_ordinal = pd.DataFrame(index=X.index)
        for col in self.ORDINAL_COLS:
            if col in X.columns:
                encoded = self.ordinal_encoders[col].fit_transform(X[[col]])
                X_ordinal[col + "_encoded"] = encoded.flatten()

        X_final = pd.concat([X_num, X_nominal, X_ordinal], axis=1)
        self.final_columns_ = X_final.columns.tolist()
        self.numeric_cols_ = numeric_cols

        y_encoded = None
        if y_raw is not None:
            y_encoded = self.label_encoder.fit_transform(y_raw)

        self.is_fitted = True
        return X_final, y_encoded

    def transform(self, df: pd.DataFrame, target_col: str = "Credit_Score"):
        if not self.is_fitted:
            raise RuntimeError("Preprocessor belum di-fit. Panggil fit() dahulu.")

        df = df.copy()
        y_raw = df[target_col] if target_col in df.columns else None
        X = df.drop(columns=[target_col]) if target_col in df.columns else df

        X = self._engineer_and_clean(X)

        X_num = self.numeric_imputer.transform(X[self.numeric_cols_])
        X_num = self.scaler.transform(X_num)
        X_num = pd.DataFrame(X_num, columns=self.numeric_cols_, index=X.index)

        for col in self.NOMINAL_COLS + self.ORDINAL_COLS:
            if col in X.columns:
                fill_val = self.cat_imputer_values_.get(col, "Unknown")
                X[col] = X[col].fillna(fill_val)

        nominal_present = [c for c in self.NOMINAL_COLS if c in X.columns]
        X_nominal = self.ohe.transform(X[nominal_present])
        nominal_feature_names = self.ohe.get_feature_names_out(nominal_present)
        X_nominal = pd.DataFrame(X_nominal, columns=nominal_feature_names, index=X.index)

        X_ordinal = pd.DataFrame(index=X.index)
        for col in self.ORDINAL_COLS:
            if col in X.columns:
                encoded = self.ordinal_encoders[col].transform(X[[col]])
                X_ordinal[col + "_encoded"] = encoded.flatten()

        X_final = pd.concat([X_num, X_nominal, X_ordinal], axis=1)
        # Pastikan kolom konsisten dengan saat fit
        X_final = X_final.reindex(columns=self.final_columns_, fill_value=0)

        y_encoded = None
        if y_raw is not None:
            y_encoded = self.label_encoder.transform(y_raw)

        return X_final, y_encoded
