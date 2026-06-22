---
title: Dengue Prediction
emoji: 🦟
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.46.0"
app_file: app.py
pinned: false
---

# 🦟 AI-Based Early Warning System for Dengue Case Prediction

> Explainable Dengue Early Warning System Using Meteorological Data and Machine Learning

---

## 📌 Overview

Proyek ini membangun sistem peringatan dini (Early Warning System/EWS) berbasis machine learning untuk memprediksi jumlah kasus demam berdarah dengue (DBD) berdasarkan data meteorologis. Model yang digunakan adalah **XGBoost Regressor** dengan tuning hyperparameter menggunakan pendekatan time series cross-validation, serta dilengkapi dengan **interpretasi model menggunakan SHAP**.

---

## 🎯 Objectives

- Memprediksi jumlah kasus dengue mingguan berdasarkan variabel cuaca
- Mengklasifikasikan tingkat risiko wabah (Normal / Alert / Outbreak / Severe Outbreak)
- Memberikan sistem peringatan dini yang dapat dijelaskan (*explainable*)

---

## 📁 Project Structure

```
dengue-prediction/
│
├── Dengue_Prediction_V3.ipynb   # Notebook utama
├── final.csv                    # Dataset input
├── dengue_xgb_model.pkl         # Model XGBoost tersimpan
├── feature_names.pkl            # Daftar nama fitur
├── ews_threshold.pkl            # Threshold EWS tersimpan
└── Dengue_EWS_Result.csv        # Output prediksi + level risiko
```

---

## 🔧 Requirements

Install dependensi berikut sebelum menjalankan notebook:

```bash
pip install xgboost catboost shap joblib scikit-learn pandas numpy matplotlib seaborn
```

Atau jalankan langsung dari dalam notebook:

```python
!pip install xgboost catboost shap joblib -q
```

---

## 📊 Dataset

File input: `final.csv`

Kolom yang digunakan sebagai fitur meteorologis:

| Fitur | Keterangan |
|---|---|
| `precip` | Curah hujan |
| `windspeed` | Kecepatan angin |
| `humidity` | Kelembaban udara |
| `temp` | Suhu udara |
| `solarradiation` | Radiasi matahari |
| `cases` | Jumlah kasus dengue (target) |

> Kolom `serial` dan `labels` dihapus pada tahap preprocessing.

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Distribusi kasus dengue
- Log transformation (`log1p`) untuk menormalisasi target
- Correlation heatmap antar variabel

### 2. Feature Engineering (Epidemiologis)
Lag features dibuat untuk menangkap efek tertunda cuaca terhadap kasus dengue:

| Variabel | Lag (minggu) |
|---|---|
| Curah hujan (`precip`) | 1, 2, 4, 8 |
| Kecepatan angin (`windspeed`) | 1, 2, 4 |
| Kelembaban (`humidity`) | 1, 2, 4, 8 |
| Suhu (`temp`) | 1, 2, 4, 8 |
| Radiasi matahari (`solarradiation`) | 1, 2, 4 |

### 3. Data Preprocessing
- Menghapus nilai kosong (NaN) akibat lag features
- Pembagian data train/test: **80% train, 20% test** (time series split)

### 4. Modelling — XGBoost Regressor
- Model dasar: `XGBRegressor(objective="reg:squarederror")`
- Hyperparameter tuning: `RandomizedSearchCV` dengan `TimeSeriesSplit(n_splits=5)`
- Parameter yang di-tuning: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`

### 5. Evaluasi Model
Prediksi dikembalikan ke skala asli menggunakan `expm1`, kemudian dievaluasi dengan:
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Koefisien Determinasi

### 6. Interpretasi Model (SHAP)
- `shap.TreeExplainer` untuk menghasilkan SHAP values
- Summary plot dan bar plot untuk feature importance

### 7. Early Warning System (EWS)
Threshold risiko ditetapkan berdasarkan statistik kasus historis:

| Level Risiko | Kondisi |
|---|---|
| 🟢 NORMAL | Prediksi < Mean |
| 🟡 ALERT | Mean ≤ Prediksi < Mean + 1 SD |
| 🔴 OUTBREAK | Mean + 1 SD ≤ Prediksi < Mean + 2 SD |
| ⚫ SEVERE OUTBREAK | Prediksi ≥ Mean + 2 SD |

---

## 📦 Output Files

| File | Keterangan |
|---|---|
| `dengue_xgb_model.pkl` | Model XGBoost terbaik hasil tuning |
| `feature_names.pkl` | Nama-nama fitur yang digunakan model |
| `ews_threshold.pkl` | Dictionary threshold EWS (normal, alert, outbreak) |
| `Dengue_EWS_Result.csv` | Hasil prediksi lengkap beserta level risiko |

---

## 🚀 How to Run

1. Pastikan file `final.csv` tersedia di direktori yang sama dengan notebook
2. Install semua dependensi
3. Jalankan notebook `Dengue_Prediction_V3.ipynb` secara berurutan dari atas ke bawah

```bash
jupyter notebook Dengue_Prediction_V3.ipynb
```

---

## 📈 Visualisasi yang Dihasilkan

- Distribusi kasus dengue (sebelum dan sesudah log transformation)
- Correlation heatmap fitur meteorologis
- Grafik Actual vs Predicted kasus dengue
- Top 20 Feature Importance (XGBoost built-in)
- SHAP Summary Plot & Bar Plot
- Distribusi level risiko EWS
- Grafik Early Warning System dengan garis threshold

---

## 🤝 Kontribusi

Pull request dan saran pengembangan sangat terbuka. Beberapa area potensial untuk pengembangan:

- Integrasi data spasial/geografis
- Penggunaan model ensemble (stacking)
- Deployment sebagai API atau dashboard interaktif
- Penambahan data kasus historis yang lebih panjang

---

## 📄 License

Proyek ini bersifat akademis dan terbuka untuk keperluan penelitian dan pengembangan kesehatan publik.
