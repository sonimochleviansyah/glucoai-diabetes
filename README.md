# GlucoAI — Skrining Risiko Diabetes (Logistic Regression)

UTS Pembelajaran Mesin. Dataset: Pima Indians Diabetes (Kaggle/UCI, NIDDK), 768 pasien, 8 fitur klinis.

## Struktur
- `index.html`, `style.css`, `script.js`: website (prediksi berjalan di browser)
- `model.json`: parameter model yang diekspor dari Python untuk website
- `diabetes.csv`, `assets/`: dataset dan grafik evaluasi
- `ml/`: kode Python
  - `UTS_Diabetes_LogReg_Colab.ipynb`: notebook Google Colab
  - `train_model.py`: script pelatihan (butuh `ml/data/diabetes.csv`)
  - `model.joblib`: model hasil latihan
  - `reports/`: metrik dan grafik

## Jalankan website lokal
```
python -m http.server 8000
```
Buka http://localhost:8000 (jangan klik dua kali index.html).

## Latih ulang model
```
cd ml
pip install pandas numpy scikit-learn joblib matplotlib
python train_model.py
```

## Deploy ke Vercel
Link deploy : https://glucoai-diabetes.vercel.app/#predict

## Hasil (data uji 20%)
Akurasi 0,734 · Presisi 0,603 · Recall 0,704 · F1 0,650 · ROC-AUC 0,813 · F1 CV 5-fold 0,669

Alat edukasi, bukan pengganti diagnosis medis.
