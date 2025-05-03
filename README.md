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

Dataset yang digunakan adalah [Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset) yang berisi:
### 📊 Jumlah Kolom Data
- Jumlah kolom: 34
- Jumlah baris: 148.670
### ⚠️ Kondisi Data
### Missing Values (Nilai Hilang)
- Ditemukan missing values pada beberapa fitur seperti LTV dan dtir1.

### Duplikat
- Tidak ditemukan baris duplikat dalam dataset.

### Outlier
Beberapa kolom numerik menunjukkan nilai ekstrem (outlier) yang tidak wajar:
- loan_amount: maksimum > 3,5 juta (rata-rata sekitar 331 ribu)
- LTV: maksimum hingga 7831.25, padahal rasio normal < 100
- income, property_value, dan dtir1: nilai tinggi ekstrem yang mencurigakan
Outlier ini perlu ditangani sebelum membangun model, baik dengan transformasi, imputasi, atau pembersihan.

### Fitur pada dataset
| Fitur                       | Tipe Data | Deskripsi Singkat                                         |
| --------------------------- | --------- | --------------------------------------------------------- |
| `ID`                        | Kategorik | Nomor identifikasi unik pinjaman                          |
| `year`                      | Numerik   | Tahun pengajuan pinjaman (selalu 2019)                    |
| `loan_limit`                | Kategorik | Jenis batas pinjaman (misal: conforming / non-conforming) |
| `Gender`                    | Kategorik | Jenis kelamin pemohon                                     |
| `approv_in_adv`             | Kategorik | Status persetujuan sebelumnya (pre-approval)              |
| `loan_type`                 | Kategorik | Tipe pinjaman (misal: type1, type2, type3)                |
| `loan_purpose`              | Kategorik | Tujuan pinjaman (misal: pembelian rumah, refinance)       |
| `Credit_Worthiness`         | Kategorik | Kelayakan kredit (l1, l2)                                 |
| `open_credit`               | Kategorik | Status kredit terbuka (ada/tidak)                         |
| `business_or_commercial`    | Kategorik | Apakah pinjaman untuk bisnis/komersial                    |
| `loan_amount`               | Numerik   | Jumlah total pinjaman                                     |
| `rate_of_interest`          | Numerik   | Suku bunga pinjaman                                       |
| `Interest_rate_spread`      | Numerik   | Selisih suku bunga terhadap standar                       |
| `Upfront_charges`           | Numerik   | Biaya yang dibayar di awal                                |
| `term`                      | Numerik   | Lama tenor pinjaman (dalam bulan)                         |
| `Neg_ammortization`         | Kategorik | Apakah terdapat amortisasi negatif                        |
| `interest_only`             | Kategorik | Apakah hanya membayar bunga                               |
| `lump_sum_payment`          | Kategorik | Apakah terdapat pembayaran sekaligus                      |
| `property_value`            | Numerik   | Nilai properti yang dibiayai                              |
| `construction_type`         | Kategorik | Jenis konstruksi properti                                 |
| `occupancy_type`            | Kategorik | Status hunian (pribadi/sewa)                              |
| `Secured_by`                | Kategorik | Jenis jaminan (contoh: rumah)                             |
| `total_units`               | Numerik   | Jumlah unit properti yang dibiayai                        |
| `income`                    | Numerik   | Pendapatan tahunan pemohon                                |
| `credit_type`               | Kategorik | Jenis institusi kredit utama (misal: CIB, EXP)            |
| `Credit_Score`              | Numerik   | Skor kredit (skala 500–900)                               |
| `co-applicant_credit_type`  | Kategorik | Jenis institusi kredit untuk co-applicant                 |
| `age`                       | Kategorik | Kelompok usia pemohon (misal: 25-34, 35-44)               |
| `submission_of_application` | Kategorik | Cara pengajuan (via institusi/perorangan)                 |
| `LTV`                       | Numerik   | Loan to Value ratio (pinjaman terhadap nilai properti)    |
| `Region`                    | Kategorik | Wilayah geografis (North, South, East, West)              |
| `Security_Type`             | Kategorik | Jenis jaminan pinjaman (direct/indirect)                  |
| `Status`                    | Numerik   | Target: Status default (0 = tidak default, 1 = default)   |
| `dtir1`                     | Numerik   | Debt-to-Income Ratio (rasio cicilan terhadap penghasilan) |


### EDA Singkat:
### 📌 Kondisi Data:
- Dataset terdiri dari 33 kolom dan sekitar 148.670 baris.
- Ditemukan missing values pada beberapa fitur seperti LTV dan dtir1.
- Tidak ditemukan duplikat data (df.duplicated().sum() menghasilkan 0).
- Beberapa fitur memiliki nilai tunggal (seperti year hanya 2019), sehingga tidak informatif.

### 🎯 Distribusi Target (Status)
Visualisasi menunjukkan distribusi kelas target sangat tidak seimbang:
- Mayoritas berstatus tidak default (Status = 0)
- Minoritas berstatus default (Status = 1)
  
![image](https://github.com/user-attachments/assets/db5ee260-89bb-4d6f-ba38-fc5423a1bd49)

📌 Insight: Model prediksi harus mempertimbangkan ketidakseimbangan ini, misalnya dengan SMOTE (oversampling) atau metode balancing lainnya agar tidak bias ke kelas mayoritas.

### 📈 Distribusi Fitur Numerik
Dilakukan visualisasi histogram untuk semua fitur numerik seperti:
- loan_amount, Credit_Score, income, LTV, Upfront_charges, dtir1, dll.

![image](https://github.com/user-attachments/assets/e7b4777f-2c01-49ef-92ff-fc92d934d217)

![image](https://github.com/user-attachments/assets/c007d2cf-cf03-44f2-b070-c607fa61a69a)

![image](https://github.com/user-attachments/assets/364abb5d-0f08-4c80-a1ce-38a3ea17951f)

![image](https://github.com/user-attachments/assets/88c78521-f412-4aa4-bc99-a5ca324db89d)

![image](https://github.com/user-attachments/assets/f0d55cd6-7801-4acf-a34b-1384f843d08a)

![image](https://github.com/user-attachments/assets/058323de-28e9-4f3d-8e98-a23f82475630)
  
📌 Temuan Penting:
- Beberapa fitur seperti LTV dan loan_amount memiliki distribusi miring kanan (positively skewed), menunjukkan adanya outlier.
- Credit_Score cenderung normal, tetapi lebih banyak terkonsentrasi di rentang 600–750.
- income menunjukkan nilai ekstrem — perlu normalisasi atau transformasi.
- dtir1 sebagian besar berada antara 20–50, yang masuk akal, namun ada outlier juga.

### heatmap korelasi fitur numerik

![image](https://github.com/user-attachments/assets/3b51618d-2152-454a-9e97-64dcab7d3cf6)

### 📌 Insight dari Heatmap Korelasi:
1. Korelasi Tinggi antar Fitur
loan_amount memiliki korelasi tinggi dengan:
- property_value (0.69) → logis, karena pinjaman biasanya proporsional terhadap nilai properti.
- income (0.44) → semakin tinggi penghasilan, semakin besar kemungkinan pinjaman yang diajukan.
- rate_of_interest dan Interest_rate_spread memiliki korelasi sangat tinggi (0.62):
Ini menunjukkan bahwa keduanya redundan dan bisa dipertimbangkan untuk memilih salah satu saja dalam model.

2. Korelasi Negatif Signifikan
- income vs dtir1: korelasi negatif sedang (-0.25) → masuk akal, karena semakin tinggi pendapatan, semakin rendah rasio utang terhadap pendapatan.
- property_value vs LTV: korelasi negatif (-0.22) → LTV (Loan-to-Value) menurun saat nilai properti meningkat
  
3. Fitur yang Tidak Berkorelasi
Beberapa fitur menunjukkan korelasi sangat rendah atau hampir nol dengan fitur lainnya, misalnya:
- Credit_Score terhadap hampir semua fitur.
- year terhadap semua fitur (seperti yang juga terlihat sebelumnya, semua data berasal dari tahun 2019).

  
## Data Preparation


Pada tahap ini dilakukan serangkaian langkah untuk mempersiapkan data sebelum digunakan dalam pelatihan model machine learning. Tahapan ini penting untuk memastikan model bekerja secara optimal dan adil dalam proses prediksi.

### 1. Train-Test Split
- Dataset dibagi menjadi data latih dan data uji menggunakan fungsi train_test_split dengan rasio 80:20:
- Tujuan: Memastikan evaluasi model dilakukan terhadap data yang belum pernah dilihat sebelumnya, sehingga hasil evaluasi lebih objektif dan realistis.

### 2. Data Preprocessing
Data preprocessing dilakukan dengan membedakan perlakuan pada fitur numerik dan kategorikal menggunakan ColumnTransformer.
- 📐 Fitur numerik: Dinormalisasi menggunakan StandardScaler agar berada pada skala yang sama.
- 🧾 Fitur kategorikal: Dikodekan dengan OrdinalEncoder untuk mengubah kategori menjadi nilai numerik.

Tujuan: Menyediakan data dalam format numerik dan berskala seragam agar dapat diproses dengan baik oleh model.

### 3. Handling Imbalanced Classes
- Masalah ketidakseimbangan kelas pada target Status diatasi menggunakan SMOTE (Synthetic Minority Oversampling Technique)
- Tujuan: Menyeimbangkan distribusi kelas target agar model tidak bias terhadap kelas mayoritas dan meningkatkan akurasi pada kelas minoritas.


## Modelling

Tahap ini berfokus pada pembangunan dan pelatihan model machine learning untuk menyelesaikan permasalahan klasifikasi status persetujuan pinjaman. Proses dilakukan melalui beberapa tahap terstruktur:

### Tahapan Pemodelan
- Preprocessing Data
> Numerik: Standarisasi menggunakan StandardScaler

> Kategorikal: Encoding dengan OrdinalEncoder

> Semua preprocessing dibungkus dalam ColumnTransformer

- Penanganan Kelas Tidak Seimbang
> Menggunakan SMOTE (Synthetic Minority Oversampling Technique) untuk memperbaiki distribusi kelas target yang tidak seimbang

- Model Training
> Dua model dilatih:

> RandomForestClassifier

> LogisticRegression

- Pipeline Machine Learning
> Pipeline dibangun dengan tahapan:

> Preprocessing → SMOTE → Model

### Random Forest Classifier
Algoritma ensemble yang membangun banyak pohon keputusan dan menggabungkannya melalui voting mayoritas untuk klasifikasi. Efektif menangani dataset besar dan kompleks.

⚙️ Parameter Awal
- random_state=42: Menjamin hasil reproducible

🔍 Kelebihan
- Mampu menangani fitur numerik dan kategorikal
- Tahan terhadap overfitting
- Menyediakan fitur feature_importance

⚠️ Kekurangan
- Kurang interpretatif
- Training time lebih lama dibanding model sederhana

### Logistic Regression
Model klasifikasi linear yang digunakan sebagai baseline. Mengestimasi probabilitas suatu kelas berdasarkan kombinasi linier fitur input.

⚙️ Parameter Awal
- max_iter=1000: Memastikan model mencapai konvergensi
- random_state=42: Konsistensi hasil

🔍 Kelebihan
- Sederhana dan cepat
- Interpretasi model mudah (koefisien dapat dibaca)
- Baik untuk hubungan linier

⚠️ Kekurangan
- Performa menurun untuk data non-linear
- Sensitif terhadap multikolinearitas

## Evaluation

Tahapan evaluasi bertujuan untuk menilai performa model machine learning yang telah dibangun menggunakan metrik-metrik evaluasi yang sesuai dengan konteks klasifikasi pada data peminjaman.

### 🎯 Evaluation Metrics
Beberapa metrik evaluasi yang digunakan:
| Metrik                       | Penjelasan                                                                                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Accuracy**                 | Proporsi prediksi yang benar dibandingkan total prediksi. Cocok jika kelas seimbang, tapi kurang informatif pada data imbalance.                 |
| **Precision**                | Kemampuan model memprediksi kelas positif secara akurat (TP / (TP + FP))                                                                         |
| **Recall**                   | Kemampuan model menangkap semua kasus positif (TP / (TP + FN))                                                                                   |
| **F1-Score**                 | Harmonic mean dari precision dan recall. Cocok untuk data tidak seimbang.                                                                        |
| **ROC-AUC Score**            | Luas area di bawah kurva ROC, menggambarkan trade-off antara TPR dan FPR.                                                                        |
| **MSE (Mean Squared Error)** | Rata-rata kuadrat selisih antara nilai prediksi dan aktual. Umumnya untuk regresi, namun digunakan di sini sebagai tambahan perspektif evaluasi. |
| **Confusion Matrix**         | Matriks yang menunjukkan TP, FP, FN, dan TN dari model prediksi.                                                                                 |

### 📐 Evaluasi Model: Mean Squared Error (MSE)
Mean Squared Error (MSE) adalah salah satu metrik evaluasi yang digunakan untuk mengukur rata-rata kuadrat selisih antara nilai prediksi dan nilai aktual. 

Formula MSE adalah:

![image](https://github.com/user-attachments/assets/f5559ee6-1662-410f-9b3d-0cc8d29d9544)

Mengapa MSE Digunakan dalam Proyek Ini:

- Meskipun MSE umum digunakan untuk regresi, pada kasus ini metrik ini digunakan sebagai indikator tambahan untuk mengevaluasi performa model klasifikasi dalam hal seberapa jauh prediksi biner model dari nilai aktual dalam bentuk kuadrat.
- Nilai MSE yang lebih kecil menunjukkan bahwa prediksi model lebih akurat terhadap nilai sebenarnya.

Interpretasi Hasil:
- Pada model Random Forest, MSE sangat kecil (mendekati 0), yang menandakan bahwa model menghasilkan prediksi yang hampir sepenuhnya akurat.
- Pada model Logistic Regression, MSE lebih besar, menandakan bahwa terdapat lebih banyak kesalahan dalam prediksi dibandingkan Random Forest.

### ✅ Random Forest - Hasil Evaluasi
- MSE: 0.000033
- Accuracy: 100%
- Classification Report:
> Precision: 1.00
> Recall: 1.00
> F1-Score: 1.00
- ROC-AUC: ~1.00
- Confusion Matrix:

  ![image](https://github.com/user-attachments/assets/a0f0d47a-d542-4694-b156-babbd11df29c)

> Interpretasi: Model sangat kuat dan mampu menangkap hampir seluruh pola dengan akurasi sempurna. Namun perlu diperhatikan kemungkinan overfitting yang tersembunyi.

### 🧮 Logistic Regression - Hasil Evaluasi
- MSE: 0.3021
- Accuracy: 70%
- Classification Report:
> Precision: 0.87 (kelas 0), 0.42 (kelas 1)

> Recall: 0.70 (kelas 0), 0.68 (kelas 1)

>F1-Score: 0.78 (kelas 0), 0.52 (kelas 1)
- ROC-AUC: ~0.70
- Confusion Matrix:

![image](https://github.com/user-attachments/assets/69f4fbfd-91e1-48c8-999c-f689bd13818b)

> Interpretasi: Sebagai baseline model, performanya masih jauh di bawah Random Forest. Terlihat precision rendah pada kelas minoritas (1).

### 🔍 ROC Curve Comparison

ROC Curve digunakan untuk mengevaluasi kemampuan model dalam membedakan antara kelas positif dan negatif. Semakin luas area di bawah kurva (AUC), semakin baik performa model.

![image](https://github.com/user-attachments/assets/bff4ea3f-301b-4e4c-819a-4965b96bea39)

Berdasarkan grafik di atas:
- **Random Forest** memiliki AUC = 1.00 → menunjukkan kinerja sangat baik.
- **Logistic Regression** memiliki AUC = 0.76 → performa cukup baik, tetapi masih kalah dibandingkan Random Forest.

### 🔁 Cross Validation
Untuk mengukur generalisasi model, dilakukan 5-fold Cross Validation:
| Model               | Mean CV Accuracy |
| ------------------- | ---------------- |
| Random Forest       | 0.99998          |
| Logistic Regression | 0.69944          |

> Hasil menunjukkan bahwa Random Forest memiliki generalisasi yang sangat tinggi dibanding Logistic Regression.

### 🔧 Hyperparameter Tuning
Dilakukan Grid Search untuk meningkatkan performa Random Forest dengan parameter:

![image](https://github.com/user-attachments/assets/20cf2018-d9c5-494b-824a-b89765e9a5f8)

- Best Parameters:

![image](https://github.com/user-attachments/assets/6b8713a1-5b8b-4d8a-8a22-422dd2fbdbeb)

- Best CV Score: 1.00

### 🏆 Model Terbaik: Random Forest Classifier
Dipilih berdasarkan:
- Skor evaluasi tertinggi (baik pada training, testing, maupun cross-validation)
- Kemampuan dalam menangani fitur yang kompleks
- Skor ROC-AUC dan F1-score sempurna

### 📊 Feature Importance (Top 10)
1. Interest_rate_spread     (0.35)
2. Upfront_charges          (0.27)
3. rate_of_interest         (0.23)
4. property_value           (0.03)
5. credit_type              (0.02)
6. LTV                      (0.02)
7. dtir1                    (0.02)
8. submission_of_application
9. co-applicant_credit_type
10. Neg_ammortization

  
---

Dokumen ini mencerminkan seluruh tahapan proyek machine learning dari pemahaman bisnis hingga evaluasi model.

Untuk tahap lanjutan, model dapat diintegrasikan ke dalam sistem scoring kredit internal perusahaan fintech, baik dalam bentuk aplikasi web atau sistem backend otomatis.

