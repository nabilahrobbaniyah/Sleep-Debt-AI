import streamlit as st

st.set_page_config(
    page_title="Sleep Debt AI Dashboard",
    layout="wide"
)

st.title("Sleep Debt AI Dashboard")
st.caption(
    "Analisis kebiasaan penggunaan smartphone pada malam hari "
    "dan prediksi kategori sleep debt menggunakan Machine Learning."
)

# daftar variabel dataset
TARGET = "sleep_debt_category"

EXCLUDED_FEATURES = [
    "user_id",
    "sleep_debt_category",
    "total_sleep_hours",
    "sleep_latency_min",
    "deep_sleep_pct",
    "rem_sleep_pct",
    "morning_alarm_snoozes",
    "next_day_fatigue_score",
]

EXPECTED_COLUMNS = [
    "user_id", "age", "gender", "occupation_type", "chronotype",
    "bedtime_phone_minutes", "primary_bedtime_app",
    "screen_brightness_pct", "blue_light_filter_active",
    "caffeine_post_5pm_mg", "physical_activity_min",
    "sleep_latency_min", "total_sleep_hours", "deep_sleep_pct",
    "rem_sleep_pct", "morning_alarm_snoozes",
    "next_day_fatigue_score", "sleep_debt_category"
]