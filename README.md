# PCOS Streamlit App

Aplikasi Streamlit untuk prediksi risiko PCOS menggunakan model final SVM kernel linear hasil Bayesian Optimization + Genetic Algorithm.

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Halaman Aplikasi

1. Beranda
2. Prediksi PCOS
   - Prediksi Manual
   - Prediksi Batch CSV
3. Informasi Model
4. Edukasi PCOS

## Fitur Prediksi Batch CSV

Gunakan tombol **Download Template CSV** pada halaman Prediksi PCOS untuk memperoleh format kolom yang sesuai dengan fitur final hasil Genetic Algorithm. Setelah file CSV diunggah, aplikasi akan menampilkan preview data, validasi kolom, hasil prediksi batch, dan tombol unduh hasil prediksi.

## Catatan

Aplikasi ini adalah prototipe penelitian dan bukan alat diagnosis medis final.
