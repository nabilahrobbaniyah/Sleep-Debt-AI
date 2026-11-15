import streamlit as st
import pandas as pd
import plotly.express as px

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

# pengaturan model
st.sidebar.header("Pengaturan Dashboard")
df = pd.read_csv("data/bedtime_screentime_sleep_debt.csv")
st.sidebar.markdown("---")
st.sidebar.subheader("Pengaturan eksperimen")

test_size = st.sidebar.slider(
    "Proporsi data testing (%)",
    min_value=10,
    max_value=40,
    value=20,
    step=5
)

random_state = st.sidebar.number_input(
    "Random state",
    min_value=0,
    max_value=9999,
    value=42
)

tree_depth = st.sidebar.slider(
    "Maksimum kedalaman Decision Tree",
    min_value=2,
    max_value=20,
    value=5
)

tree_min_samples = st.sidebar.slider(
    "Minimum sampel pada leaf Decision Tree",
    min_value=1,
    max_value=20,
    value=5
)

# validasi dataset
st.subheader("1. Dataset Overview")

if TARGET not in df.columns:
    st.error(
        f"Kolom target '{TARGET}' tidak ditemukan. "
        "Periksa kembali nama kolom pada file CSV."
    )
    st.write("Kolom yang tersedia:", df.columns.tolist())
    st.stop()

df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = df[TARGET].astype(str).str.strip()

# ringkasan metrik
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah responden", f"{len(df):,}")
c2.metric("Jumlah kolom", len(df.columns))
c3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
c4.metric("Jumlah kategori target", df[TARGET].nunique())

with st.expander("Lihat data mentah"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Lihat informasi tipe data dan missing values"):
    info_df = pd.DataFrame({
        "Tipe Data": df.dtypes.astype(str),
        "Missing Values": df.isna().sum(),
        "Missing (%)": (df.isna().mean() * 100).round(2)
    })
    st.dataframe(info_df, use_container_width=True)

# grafik distribusi kategori target
target_counts = df[TARGET].value_counts().reset_index()
target_counts.columns = ["Kategori Sleep Debt", "Jumlah"]

fig_target = px.bar(
    target_counts,
    x="Kategori Sleep Debt",
    y="Jumlah",
    color="Kategori Sleep Debt",
    title="Distribusi Kategori Sleep Debt",
    text="Jumlah"
)
st.plotly_chart(fig_target, use_container_width=True)