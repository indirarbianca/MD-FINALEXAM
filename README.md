# UAS Model Deployment — Credit Score Classification

**Nama:** Bian
**NIM:** 2802534166

## Deskripsi
Proyek ini membangun model machine learning untuk memprediksi kategori credit
score nasabah (`Good` / `Standard` / `Poor`) berdasarkan data finansial dan
riwayat pembayaran, lengkap dengan pipeline training ber-MLflow, deployment
Streamlit lokal, dan rencana deployment cloud (AWS).

## Struktur File

| File / Folder | Deskripsi | Poin Soal |
|---|---|---|
| `2802534166_EDA_Modeling.ipynb` | Notebook eksplorasi data & eksperimen modeling (4 model dibandingkan) | 1a |
| `pipeline.py` | Pipeline training OOP dengan MLflow tracking | 1b |
| `preprocessing.py` | Class `CreditDataPreprocessor` (dependency wajib untuk pipeline & inference) | 1b, 1c |
| `best_model.pkl` | Model terbaik (Random Forest) + preprocessor terlatih | 1b, 1c |
| `inference.py` | Script inferencing (dipakai oleh app.py & bisa dites lewat CLI) | 1c |
| `app.py` | Aplikasi Streamlit untuk deployment | 1c |
| `requirements.txt` | Daftar dependency Python | 1c |
| `2802534166_Screenshot_Test_Deployment.pdf` | Template/hasil screenshot testing (lihat catatan di bawah) | 1c, 2b |
| `fill_pdf_with_screenshots.py` | Script untuk mengisi otomatis PDF dengan screenshot asli | 1c, 2b |
| `TESTING_GUIDE.md` | Panduan & 3 test case (Good/Standard/Poor) untuk testing deployment | 1c, 2b |
| `aws/` | Script & panduan pipeline + deployment berbasis AWS (SageMaker, EC2) | 2a, 2b |
| `aws/AWS_DEPLOYMENT_GUIDE.md` | Termasuk perbandingan local vs cloud | 2a, 2b, 3 |
| `VIDEO_SCRIPT.md` | Kerangka talking points untuk video penjelasan | 4 |
| `LINK_VIDEO.txt` | Tempat mencantumkan link video setelah direkam | 4 |

> **Catatan penamaan file:** `pipeline.py`, `preprocessing.py`, dan `inference.py`
> sengaja **tidak** diberi prefix NIM karena nama file Python yang diawali angka
> tidak bisa di-import sebagai modul (`from inference import ...` akan gagal
> jika file bernama `2802534166_inference.py`). File-file ini tetap merupakan
> jawaban untuk poin 1b dan 1c/4b/4d sesuai ketentuan, hanya penamaannya
> disesuaikan agar kode tetap berfungsi.

## Cara Menjalankan

### 1. Eksplorasi & Modeling
Buka `2802534166_EDA_Modeling.ipynb` di Jupyter Notebook / JupyterLab / Google Colab.

### 2. Training Pipeline (lokal)
```bash
pip install -r requirements.txt
pip install mlflow xgboost
python pipeline.py
```
Hasil tracking bisa dilihat dengan: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
lalu buka `http://localhost:5000`.

### 3. Testing Inferencing (CLI)
```bash
python inference.py
```

### 4. Deployment Streamlit (lokal)
```bash
streamlit run app.py
```
Buka `http://localhost:8501` di browser.

### 5. Deployment Streamlit (cloud / public URL)
Lihat panduan lengkap di `aws/AWS_DEPLOYMENT_GUIDE.md`.

## Catatan Penting — Screenshot Testing
File `2802534166_Screenshot_Test_Deployment.pdf` yang disertakan adalah
**template** berisi instruksi dan placeholder kosong, karena pengambilan
screenshot memerlukan aplikasi yang dijalankan secara interaktif di browser
oleh pengguna (tidak bisa di-generate otomatis di sisi pembuatan kode).

**Langkah melengkapi sebelum submit:**
1. Jalankan aplikasi (lokal & cloud), lakukan 3 testing sesuai `TESTING_GUIDE.md`.
2. Screenshot tiap hasil, simpan dengan nama: `local_good.png`, `local_standard.png`,
   `local_poor.png`, `cloud_good.png`, `cloud_standard.png`, `cloud_poor.png`.
3. Jalankan `python fill_pdf_with_screenshots.py` (sudah berisi screenshot asli) →
   menghasilkan `Screenshot_Test_Deployment.pdf` final yang siap dilampirkan
   (ganti file template di poin e dengan hasil ini).

## Ringkasan Hasil Eksperimen

| Model | Accuracy | F1-macro |
|---|---|---|
| **Random Forest** (terbaik) | 0.7322 | **0.7105** |
| Random Forest (tuned) | 0.7274 | 0.7069 |
| XGBoost (tuned) | 0.7276 | 0.7057 |
| XGBoost | 0.7126 | 0.6903 |
| KNN | 0.6510 | 0.6192 |
| Logistic Regression | 0.6322 | 0.5945 |
