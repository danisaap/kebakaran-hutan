# Dashboard Hotspot Kebakaran Hutan di Provinsi Jawa Timur

Dashboard interaktif berbasis **Streamlit** untuk memvisualisasikan dan menganalisis titik hotspot kebakaran hutan yang terdeteksi satelit di Provinsi Jawa Timur.

## Fitur

- **Dashboard Peta Interaktif** — Peta sebaran hotspot dengan marker berdasarkan tingkat confidence (High/Medium/Low), filter tanggal, heatmap, dan perbandingan dengan data kebakaran aktual.
- **Visualisasi Data** — Grafik interaktif sebaran hotspot per kabupaten, time series, distribusi confidence, sebaran geografis, dan analisis waktu kejadian.
- **Hasil Clustering** — Tampilan hasil clustering K-Means dan dendrogram hierarchical clustering (Ward) untuk mengelompokkan kabupaten berdasarkan pola hotspot dan luas kebakaran.
- **Forecasting & Anomali** — Prediksi jumlah hotspot bulanan menggunakan model SARIMA dan deteksi anomali dengan metode Z-Score.

## Dataset

- Data hotspot satelit NASA tahun 2021–2025 (5 file CSV)
- Data kebakaran aktual (karhut) tahun 2021–2023
- Hasil clustering K-Means dan model forecasting (pre-generated)

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run stflame.py
```

## Struktur File

| File | Deskripsi |
|------|-----------|
| `stflame.py` | Aplikasi utama Streamlit |
| `requirements.txt` | Daftar dependensi Python |
| `model_kmeans_hotspot.pkl` | Model K-Means terlatih |
| `scaler_hotspot.pkl` | Scaler untuk normalisasi fitur |
| `hasil_cluster_hotspot.csv` | Hasil clustering per kabupaten |
| `forecast_hotspot.csv` | Hasil forecasting SARIMA 12 bulan |
| `histori_hotspot.csv` | Histori hotspot bulanan untuk forecasting |
| `anomali_hotspot.csv` | Hasil deteksi anomali per kabupaten |
