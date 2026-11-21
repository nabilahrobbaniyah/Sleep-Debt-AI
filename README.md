# Final project — Artificial Intelligence Subject
## Menjalankan aplikasi

1. Install Python 3.10 atau lebih baru
2. Buka terminal pada folder proyek
3. Buat virtual environment:
   `python -m venv venv`
4. Aktifkan di Windows:
   `venv\\Scripts\\activate`
5. Install library:
   `pip install -r requirements.txt`
6. Jalankan:
   `streamlit run app.py`
7. Unggah file `bedtime_screentime_sleep_debt.csv` melalui sidebar

## Catatan metodologi

Target: `sleep_debt_category`

Fitur yang dikeluarkan dari prediktor untuk mengurangi risiko target leakage:
`user_id`, `total_sleep_hours`, `sleep_latency_min`, `deep_sleep_pct`,
`rem_sleep_pct`, `morning_alarm_snoozes`, `next_day_fatigue_score`.

Aplikasi membandingkan Logistic Regression dan Decision Tree Classifier
