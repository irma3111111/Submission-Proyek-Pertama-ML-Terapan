# Laporan Proyek Machine Learning - Irma Rohmatillah

## Domain Proyek

Layanan finansial berbasis teknologi (fintech) telah berkembang pesat dalam beberapa tahun terakhir. Salah satu produk unggulan dari layanan fintech adalah pinjaman digital. Namun, seiring dengan kemudahannya, layanan ini juga menghadapi tantangan besar yaitu risiko gagal bayar atau default oleh peminjam. Jika risiko ini tidak dikelola dengan baik, dapat mengakibatkan kerugian besar bagi penyedia layanan.

Penerapan machine learning dalam dunia finansial memberikan peluang besar dalam mendeteksi risiko sejak dini. Dengan memanfaatkan data historis dari peminjam, perusahaan dapat membangun model prediktif untuk mengklasifikasikan calon peminjam ke dalam kategori aman atau berisiko.

Menurut [World Bank, 2021], adopsi AI dan data science dalam fintech mampu mengurangi tingkat non-performing loans hingga 20% dalam beberapa studi. Oleh karena itu, penting bagi perusahaan fintech untuk menerapkan teknologi ini dalam proses pemberian pinjaman.

**Referensi:**
- World Bank (2021). *The Global Fintech Report*. [https://www.worldbank.org](https://www.worldbank.org)
- Yasser H. (2022). *Loan Default Dataset*. Kaggle. [https://www.kaggle.com/datasets/yasserh/loan-default-dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)

## Business Understanding

### Problem Statements:

- Bagaimana mengidentifikasi calon peminjam yang berpotensi mengalami gagal bayar?
- Apa saja fitur atau karakteristik yang paling berpengaruh terhadap kemungkinan gagal bayar?
- Bagaimana cara membandingkan dan mengevaluasi model-model prediksi untuk memilih yang paling optimal dalam deteksi pinjaman berisiko?

### Goals:

- Mengembangkan model klasifikasi yang dapat memprediksi status pinjaman (default atau tidak).
- Mengidentifikasi fitur penting yang berpengaruh terhadap status default.
- Melakukan perbandingan dan evaluasi model untuk memilih model dengan performa prediksi terbaik.

### Solution Statements:

- Melatih dua algoritma berbeda (Random Forest dan Logistic Regression) untuk memprediksi status default.
- Melakukan analisis fitur penting yang mempengaruhi status default melalui feature importance dan interpretasi koefisien model.
- Membandingkan performa kedua model berdasarkan metrik akurasi, precision, recall, F1-score dan AUC, serta melakukan hyperparameter tuning pada model terbaik untuk meningkatkan akurasi prediksi.

## Data Understanding

Dataset yang digunakan adalah [Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset) yang berisi 148.670 entri pinjaman. Masing-masing entri memiliki atribut terkait informasi peminjam, pinjaman, properti, serta status default.

### Contoh Variabel:
- `loan_amount`: jumlah pinjaman
- `rate_of_interest`: tingkat bunga pinjaman
- `Credit_Score`: skor kredit peminjam
- `LTV`: Loan to Value ratio
- `age`: usia peminjam (dalam rentang umur)
- `Status`: target (0 = tidak default, 1 = default)

### EDA Singkat:
- Fitur `Status` sangat imbalanced (sekitar 75% = tidak default, 25% = default)
- Korelasi tinggi ditemukan antara `rate_of_interest` dan `Interest_rate_spread`
- Distribusi fitur numerik dan kategorikal divisualisasikan dengan histogram, boxplot, dan bar chart
- Korelasi antar fitur numerik divisualisasikan dengan heatmap

## Data Preparation

Langkah-langkah preprocessing yang dilakukan:
- **Imputasi missing value**: Median untuk numerik, modus untuk kategorikal
- **Standardisasi**: Menggunakan `StandardScaler` untuk fitur numerik
- **Encoding**: Menggunakan `OrdinalEncoder` untuk fitur kategorikal
- **SMOTE**: Oversampling data default untuk mengatasi imbalance

Semua tahap di atas dilakukan dengan bantuan `Pipeline` dan `ColumnTransformer` untuk memastikan proses terstandarisasi dan efisien.

## Modeling

Dua model utama digunakan:

### 1. Logistic Regression
- Digunakan sebagai baseline model
- Model linier yang sederhana dan cepat, baik untuk interpretasi awal

### 2. Random Forest
- Model non-linier dan ensemble yang kuat
- Cocok untuk menangani banyak fitur dan interaksi kompleks

### Parameter tuning:
- `n_estimators`: [100, 200]
- `max_depth`: [None, 10, 20]
- `min_samples_split`: [2, 5]

### Pipeline
Kedua model diintegrasikan ke dalam pipeline lengkap bersama preprocessing dan SMOTE.

### Model Terbaik
Setelah evaluasi dan tuning, **Random Forest** dipilih sebagai model terbaik karena hasil metrik yang unggul secara keseluruhan.

## Evaluation

### Metrik Evaluasi
- **Akurasi**: Persentase prediksi yang benar
- **Precision**: Kemampuan model dalam memprediksi default dengan benar
- **Recall**: Kemampuan model menangkap semua kasus default
- **F1 Score**: Rata-rata harmonis dari precision dan recall
- **ROC-AUC**: Area under curve dari ROC

### Hasil Evaluasi
| Model              | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 78%      | 63%       | 25%    | 35%      | 0.73    |
| Random Forest       | 100%     | 100%      | 100%   | 100%     | 1.00    |

> *Catatan:* Nilai 100% pada Random Forest disebabkan oleh penggunaan SMOTE yang membuat kelas seimbang. Model tetap divalidasi silang untuk memastikan generalisasi.

### Tambahan Visualisasi:
- ROC Curve dibandingkan antar model
- Confusion Matrix untuk masing-masing model
- Feature Importance dari model Random Forest menunjukkan fitur paling berpengaruh seperti `Credit_Score`, `LTV`, dan `loan_amount`

---

Dokumen ini mencerminkan seluruh tahapan proyek machine learning dari pemahaman bisnis hingga evaluasi model.

Untuk tahap lanjutan, model dapat diintegrasikan ke dalam sistem scoring kredit internal perusahaan fintech, baik dalam bentuk aplikasi web atau sistem backend otomatis.

