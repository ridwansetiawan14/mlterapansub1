# Laporan Proyek Machine Learning - Ridwan Setiawan (abu_akhdan)

## **Domain Proyek**

Proyek ini berfokus pada bidang **pendidikan tinggi**, khususnya analisis risiko putus kuliah (dropout) mahasiswa di awal masa studi. Dengan meningkatnya jumlah mahasiswa yang tidak menyelesaikan studinya tepat waktu atau bahkan keluar sebelum lulus, terdapat kebutuhan mendesak untuk membangun sistem prediktif yang dapat mendeteksi mahasiswa yang berpotensi mengalami masalah akademik sejak dini[1,2,3].

Studi ini menggunakan data yang dikumpulkan dari universitas di Portugal yang mencakup informasi demografis, akademik, serta sosial ekonomi mahasiswa pada saat awal pendaftaran. Tujuan dari proyek ini adalah memprediksi status akhir mahasiswa—dropout, masih aktif (enrolled), atau lulus (graduate)—dengan pendekatan pembelajaran mesin (machine learning).

Sumber data: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success  
Referensi utama: 
[1] Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M., & Realinho, V. (2021). Early prediction of student’s performance in higher education: A case study. In Trends and Applications in Information Systems and Technologies: Volume 1 9 (pp. 166-175). Springer International Publishing.
[2] Kok, C. L., Ho, C. K., Chen, L., Koh, Y. Y., & Tian, B. (2024). A Novel Predictive Modeling for Student Attrition Utilizing Machine Learning and Sustainable Big Data Analytics. Applied Sciences, 14(21), 9633.
[3] Martins, M. V., Baptista, L., Machado, J., & Realinho, V. (2023). Multi-class phased prediction of academic performance and dropout in higher education. Applied Sciences, 13(8), 4702.

## **Business Understanding**

Putus kuliah di tingkat pendidikan tinggi merupakan isu serius yang berdampak luas, baik secara institusional, personal, maupun sosial. Di banyak negara, termasuk Portugal sebagai lokasi dataset ini, angka dropout yang tinggi menyebabkan kerugian finansial bagi institusi pendidikan, meningkatnya beban ekonomi keluarga, serta rendahnya capaian pendidikan nasional.

Melalui pendekatan data-driven, kita dapat membangun sistem peringatan dini untuk mengidentifikasi mahasiswa yang berisiko tinggi tidak menyelesaikan studi. Deteksi dini ini memungkinkan intervensi personal yang lebih tepat sasaran, seperti bimbingan akademik atau dukungan finansial.

### **Problem Statements**

1. Bagaimana memprediksi status akhir mahasiswa (dropout, enrolled, graduate) hanya dari data saat pendaftaran?
2. Fitur-fitur mana yang paling memengaruhi kemungkinan seorang mahasiswa mengalami dropout?
3. Seberapa baik model machine learning dapat mengklasifikasikan status akhir mahasiswa dalam skenario multikelas yang tidak seimbang?

### **Goals**

1. Mengembangkan model prediktif berbasis data awal mahasiswa untuk memetakan kemungkinan status akhir studi.
2. Melakukan analisis korelasi dan eksploratif untuk mengidentifikasi fitur paling signifikan terhadap status kelulusan.
3. Mengevaluasi performa model dalam skenario multikelas dan dataset yang imbalanced, serta melakukan balancing dengan SMOTE-Tomek.

### **Solution Statements**

Untuk menjawab tujuan di atas, pendekatan berikut diterapkan:

- Melakukan preprocessing dan EDA secara menyeluruh terhadap data pendaftaran mahasiswa.
- Menggunakan teknik balancing data (SMOTE-Tomek) untuk memperbaiki distribusi label yang tidak seimbang.
- Membangun beberapa model klasifikasi (Logistic Regression, SVM, Decision Tree, Random Forest, XGBoost) dan membandingkan performanya berdasarkan metrik klasifikasi multikelas (accuracy, precision, recall, F1-score).
- Menggunakan hyperparameter tuning untuk meningkatkan akurasi model terbaik, serta interpretabilitas melalui visualisasi fitur penting.


## **Data Understanding**

Dataset yang digunakan dalam proyek ini diperoleh dari repositori UCI Machine Learning yang berjudul [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/).dan berisi 4.424 data mahasiswa yang bersumber dari `perguruan tinggi di portugal`. Masing-masing baris mewakili satu mahasiswa dengan total 37 atribut, termasuk informasi demografis, akademik, sosial ekonomi, serta label status akhir mahasiswa (Target).

### **Deskripsi Fitur**

Dataset ini memiliki 36 fitur input dan 1 fitur target. Berikut adalah daftar lengkap fitur:

1. **Marital status:** Status pernikahan mahasiswa
2. **Application mode:** Jalur pendaftaran yang digunakan
3. **Application order:** Urutan pilihan program studi
4. **Course:** Kode program studi yang dipilih
5. **Daytime/evening attendance:** Waktu kehadiran kuliah (siang/malam)
6. **Previous qualification:** Jenis kualifikasi sebelum masuk kuliah
7. **Previous qualification (grade):** Nilai pada kualifikasi sebelumnya
8. **Nacionality:** Kode kewarganegaraan mahasiswa
9. **Mother's qualification:** Tingkat pendidikan ibu
10. **Father's qualification:** Tingkat pendidikan ayah
11. **Mother's occupation:** Jenis pekerjaan ibu
12. **Father's occupation:** Jenis pekerjaan ayah
13. **Admission grade:** Nilai akhir saat diterima di universitas
14. **Displaced:** Status apakah mahasiswa tinggal jauh dari rumah asal
15. **Educational special needs:** Status kebutuhan khusus pendidikan
16. **Debtor:** Status apakah mahasiswa memiliki tunggakan
17. **Tuition fees up to date:** Apakah pembayaran biaya kuliah lancar
18. **Gender:** Jenis kelamin mahasiswa
19. **Scholarship holder:** Status kepemilikan beasiswa
20. **Age at enrollment:** Usia saat pendaftaran
21. **International:** Status internasional mahasiswa
22. **Curricular units 1st sem (credited):** Mata kuliah yang dikonversi semester 1
23. **Curricular units 1st sem (enrolled):** Mata kuliah yang diambil semester 1
24. **Curricular units 1st sem (evaluations):** Evaluasi yang dilakukan semester 1
25. **Curricular units 1st sem (approved):** Mata kuliah yang lulus semester 1
26. **Curricular units 1st sem (grade):** Nilai rata-rata semester 1
27. **Curricular units 1st sem (without evaluations):** Mata kuliah tanpa evaluasi semester 1
28. **Curricular units 2nd sem (credited):** Mata kuliah yang dikonversi semester 2
29. **Curricular units 2nd sem (enrolled):** Mata kuliah yang diambil semester 2
30. **Curricular units 2nd sem (evaluations):** Evaluasi yang dilakukan semester 2
31. **Curricular units 2nd sem (approved):** Mata kuliah yang lulus semester 2
32. **Curricular units 2nd sem (grade):** Nilai rata-rata semester 2
33. **Curricular units 2nd sem (without evaluations):** Mata kuliah tanpa evaluasi semester 2
34. **Unemployment rate:** Tingkat pengangguran saat itu
35. **Inflation rate:** Tingkat inflasi saat itu
36. **GDP:** Produk Domestik Bruto saat itu

Fitur ke-37 adalah:
- **Target:** Kategori status mahasiswa akhir (0 = Dropout, 1 = Enrolled, 2 = Graduate)

Seluruh fitur tidak memiliki missing values, dan sebagian besar merupakan data numerik atau dikodekan sebagai angka kategorik.

### **Distribusi Kelas (Target)**

Berikut merupakan distribusi target (kelas) yang ada pada dataset ditampilkan pada gambar berikut:


Distribusi label target menunjukkan ketidakseimbangan kelas yang signifikan:

| Label | Jumlah |
|-------|--------|
| Dropout   | 1421   |
| Enrolled  | 794    |
| Graduate  | 2209   |

Distribusi ini memperlihatkan bahwa kelas "Graduate" merupakan mayoritas, sedangkan "Enrolled" merupakan kelas minoritas yang paling sedikit jumlahnya.

### 3.3 Statistik Deskriptif

Statistik deskriptif untuk variabel numerik menunjukkan adanya variasi yang cukup besar, terutama pada fitur nilai akademik dan jumlah mata kuliah. Hal ini memberikan indikasi awal bahwa beberapa fitur memiliki potensi prediktif yang kuat.

### 3.4 Korelasi Antar Fitur

Analisis korelasi Spearman dipilih karena lebih robust terhadap data non-linear dan ordinal. Hasil analisis menunjukkan bahwa fitur-fitur berikut memiliki korelasi signifikan terhadap status akhir mahasiswa:

- Curricular units 2nd sem (approved): 0.62
- Curricular units 2nd sem (grade): 0.56
- Curricular units 1st sem (approved): 0.52
- Curricular units 1st sem (grade): 0.48
- Tuition fees up to date: 0.41
- Scholarship holder: 0.29
- Age at enrollment: -0.24
- Gender: -0.22
- Debtor: -0.24
- Curricular units 2nd sem (enrolled): 0.17
- Admission grade: 0.16

### 3.5 Visualisasi Fitur Utama

Visualisasi dilakukan untuk masing-masing fitur yang memiliki korelasi > 0.20 atau < -0.20 terhadap target. Diagram yang digunakan meliputi pie chart, histogram dengan KDE, dan boxplot terhadap target. Visualisasi ini mengungkapkan pola yang berbeda antara mahasiswa yang dropout, masih aktif, atau sudah lulus.

### 3.6 Visualisasi Pairwise dan Distribusi Numerik

Visualisasi tambahan menggunakan pairplot dan histogram terpisah menunjukkan bahwa sebagian besar fitur numerik memiliki distribusi yang tidak simetris dan terdapat perbedaan mencolok antar kelas, terutama pada performa akademik mahasiswa.















Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

