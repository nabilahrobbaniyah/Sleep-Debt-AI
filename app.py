import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

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

# EDA
st.markdown("---")
st.subheader("2. Exploratory Data Analysis")

tab1, tab2, tab3 = st.tabs([
    "Penggunaan Smartphone",
    "Durasi Tidur",
    "Korelasi Variabel"
])

with tab1:
    if "bedtime_phone_minutes" in df.columns:
        fig_phone = px.histogram(
            df,
            x="bedtime_phone_minutes",
            nbins=30,
            title="Distribusi Durasi Penggunaan Smartphone Sebelum Tidur",
            labels={"bedtime_phone_minutes": "Durasi (menit)"}
        )
        st.plotly_chart(fig_phone, use_container_width=True)

    if {"bedtime_phone_minutes", TARGET}.issubset(df.columns):
        fig_box = px.box(
            df,
            x=TARGET,
            y="bedtime_phone_minutes",
            color=TARGET,
            title="Durasi Penggunaan Smartphone Berdasarkan Kategori Sleep Debt"
        )
        st.plotly_chart(fig_box, use_container_width=True)

with tab2:
    if "total_sleep_hours" in df.columns:
        fig_sleep = px.histogram(
            df,
            x="total_sleep_hours",
            nbins=30,
            title="Distribusi Total Durasi Tidur",
            labels={"total_sleep_hours": "Durasi tidur (jam)"}
        )
        st.plotly_chart(fig_sleep, use_container_width=True)

    if {"bedtime_phone_minutes", "total_sleep_hours"}.issubset(df.columns):
        fig_scatter = px.scatter(
            df,
            x="bedtime_phone_minutes",
            y="total_sleep_hours",
            color=TARGET,
            opacity=0.6,
            title="Penggunaan Smartphone dan Durasi Tidur",
            labels={
                "bedtime_phone_minutes": "Penggunaan smartphone (menit)",
                "total_sleep_hours": "Durasi tidur (jam)"
            }
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Belum cukup variabel numerik untuk membuat heatmap")

# menentukan fitur dan target
st.markdown("---")
st.subheader("3. Persiapan Machine Learning")

feature_cols = [
    col for col in df.columns
    if col not in EXCLUDED_FEATURES
]

if len(feature_cols) == 0:
    st.error("Tidak ada fitur prediktor yang dapat digunakan")
    st.stop()

X = df[feature_cols].copy()
y = df[TARGET].copy()

# hapus fitur yang seluruh nilainya kosong
X = X.dropna(axis=1, how="all")
feature_cols = X.columns.tolist()

st.write("**Variabel prediktor yang digunakan:**")
st.write(feature_cols)
st.write("**Variabel target:**", TARGET)

# menentukan kolom numerik dan kategorikal secara otomatis
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

# memastikan kelas target cukup untuk stratified split
class_counts = y.value_counts()
if class_counts.min() < 2:
    st.error(
        "Setidaknya satu kategori target hanya memiliki satu data. "
        "Stratified train-test split tidak dapat dilakukan."
    )
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size / 100,
    random_state=int(random_state),
    stratify=y
)

st.write(f"Jumlah data training: **{len(X_train):,}**")
st.write(f"Jumlah data testing: **{len(X_test):,}**")

# preprocessing dan pipeline model
numeric_preprocess_lr = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

numeric_preprocess_tree = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_preprocess = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor_lr = ColumnTransformer(transformers=[
    ("num", numeric_preprocess_lr, numeric_features),
    ("cat", categorical_preprocess, categorical_features)
])

preprocessor_tree = ColumnTransformer(transformers=[
    ("num", numeric_preprocess_tree, numeric_features),
    ("cat", categorical_preprocess, categorical_features)
])
# buat 2 model klasifikasi
logistic_model = Pipeline(steps=[
    ("preprocessor", preprocessor_lr),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

decision_tree_model = Pipeline(steps=[
    ("preprocessor", preprocessor_tree),
    ("classifier", DecisionTreeClassifier(
        max_depth=tree_depth,
        min_samples_leaf=tree_min_samples,
        random_state=int(random_state),
        class_weight="balanced"
    ))
])

# training dan evaluasi model

# Tombol digunakan agar proses training dijalankan saat diminta.
if st.button("Jalankan Training dan Evaluasi Model", type="primary"):

    with st.spinner("Melatih Logistic Regression dan Decision Tree..."):

        # Melatih kedua model dengan data training yang sama.
        logistic_model.fit(X_train, y_train)
        decision_tree_model.fit(X_train, y_train)

        # Prediksi terhadap data testing yang belum dilihat model.
        pred_lr = logistic_model.predict(X_test)
        pred_dt = decision_tree_model.predict(X_test)

        # Menghitung metrik evaluasi.
        def evaluate_model(name, y_true, y_pred):
            return {
                "Model": name,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision (Macro)": precision_score(
                    y_true, y_pred, average="macro", zero_division=0
                ),
                "Recall (Macro)": recall_score(
                    y_true, y_pred, average="macro", zero_division=0
                ),
                "F1-Score (Macro)": f1_score(
                    y_true, y_pred, average="macro", zero_division=0
                )
            }

        results = pd.DataFrame([
            evaluate_model("Logistic Regression", y_test, pred_lr),
            evaluate_model("Decision Tree", y_test, pred_dt)
        ])

        st.session_state["results"] = results
        st.session_state["pred_lr"] = pred_lr
        st.session_state["pred_dt"] = pred_dt
        st.session_state["y_test"] = y_test
        st.session_state["logistic_model"] = logistic_model
        st.session_state["decision_tree_model"] = decision_tree_model
        st.session_state["X_test"] = X_test
        st.session_state["feature_cols"] = feature_cols

    st.success("Training dan evaluasi kedua model selesai.")
    
# menampilkan perbandingan model
if "results" in st.session_state:

    st.markdown("---")
    st.subheader("4. Model Performance Comparison")

    results = st.session_state["results"]

    st.dataframe(
        results.style.format({
            "Accuracy": "{:.3f}",
            "Precision (Macro)": "{:.3f}",
            "Recall (Macro)": "{:.3f}",
            "F1-Score (Macro)": "{:.3f}"
        }),
        use_container_width=True
    )

    # Grafik perbandingan performa.
    results_long = results.melt(
        id_vars="Model",
        var_name="Metrik",
        value_name="Skor"
    )

    fig_metrics = px.bar(
        results_long,
        x="Metrik",
        y="Skor",
        color="Model",
        barmode="group",
        text_auto=".3f",
        title="Perbandingan Performa Logistic Regression dan Decision Tree"
    )
    fig_metrics.update_yaxes(range=[0, 1])
    st.plotly_chart(fig_metrics, use_container_width=True)

    # Model dengan macro F1-score tertinggi.
    best_model = results.loc[
        results["F1-Score (Macro)"].idxmax(), "Model"
    ]
    st.info(
        f"Model dengan Macro F1-Score tertinggi pada pengujian ini: "
        f"*{best_model}*. Hasil ini berlaku untuk pembagian data dan "
        "pengaturan eksperimen yang digunakan."
    )
    
# confusion matrix
st.subheader("5. Confusion Matrix")

    labels = sorted(y.unique().tolist())

    col_a, col_b = st.columns(2)

    for col, model_name, prediction in [
        (col_a, "Logistic Regression", st.session_state["pred_lr"]),
        (col_b, "Decision Tree", st.session_state["pred_dt"])
    ]:
        cm = confusion_matrix(
            st.session_state["y_test"],
            prediction,
            labels=labels
        )

        fig_cm = px.imshow(
            cm,
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
            labels={
                "x": "Prediksi Model",
                "y": "Kategori Aktual",
                "color": "Jumlah"
            },
            title=f"Confusion Matrix - {model_name}"
        )
        col.plotly_chart(fig_cm, use_container_width=True)

# explainable ai sederhana
st.markdown("---")
    st.subheader("6. Explainable AI")

    st.write(
        "Bagian ini menunjukkan fitur yang paling banyak digunakan "
        "oleh model dalam membuat prediksi. Feature importance bukan "
        "bukti hubungan sebab-akibat."
    )

    trained_tree = st.session_state["decision_tree_model"]
    trained_classifier = trained_tree.named_steps["classifier"]
    trained_preprocessor = trained_tree.named_steps["preprocessor"]

    # Mengambil nama fitur setelah proses One-Hot Encoding.
    try:
        transformed_names = trained_preprocessor.get_feature_names_out()
        importances = trained_classifier.feature_importances_

        importance_df = pd.DataFrame({
            "Fitur": transformed_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(15)

        fig_importance = px.bar(
            importance_df.sort_values("Importance"),
            x="Importance",
            y="Fitur",
            orientation="h",
            title="15 Fitur Terpenting - Decision Tree"
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    except Exception as error:
        st.warning(f"Feature importance belum dapat ditampilkan: {error}")

    # Menampilkan koefisien Logistic Regression.
    st.markdown("*Koefisien Logistic Regression*")
    trained_lr = st.session_state["logistic_model"]
    lr_classifier = trained_lr.named_steps["classifier"]
    lr_preprocessor = trained_lr.named_steps["preprocessor"]

    try:
        lr_names = lr_preprocessor.get_feature_names_out()
        coefficients = lr_classifier.coef_

        # Untuk multiclass, hitung rata-rata nilai absolut koefisien
        # antar kelas sebagai ringkasan besarnya koefisien.
        mean_abs_coef = np.abs(coefficients).mean(axis=0)

        coef_df = pd.DataFrame({
            "Fitur": lr_names,
            "Rata-rata |Koefisien|": mean_abs_coef
        }).sort_values(
            "Rata-rata |Koefisien|", ascending=False
        ).head(15)

        fig_coef = px.bar(
            coef_df.sort_values("Rata-rata |Koefisien|"),
            x="Rata-rata |Koefisien|",
            y="Fitur",
            orientation="h",
            title="15 Fitur dengan Koefisien Absolut Terbesar"
        )
        st.plotly_chart(fig_coef, use_container_width=True)
    except Exception as error:
        st.warning(f"Koefisien belum dapat ditampilkan: {error}")

# simulator prediksi
st.markdown("---")
    st.subheader("7. Prediction Simulator")

    st.write(
        "Masukkan karakteristik individu untuk mencoba prediksi "
        "kategori sleep debt. Nilai yang dimasukkan harus mengikuti "
        "rentang dan kategori yang ada pada dataset."
    )

    input_values = {}

    with st.form("prediction_form"):

        input_columns = st.columns(2)

        for index, feature in enumerate(feature_cols):
            col = input_columns[index % 2]

            if feature in numeric_features:
                # Nilai awal menggunakan median data training.
                default_value = float(
                    pd.to_numeric(X_train[feature], errors="coerce").median()
                )
                input_values[feature] = col.number_input(
                    feature,
                    value=default_value
                )
            else:
                # Pilihan kategori diambil dari data yang tersedia.
                options = sorted(
                    X_train[feature].dropna().astype(str).unique().tolist()
                )
                input_values[feature] = col.selectbox(
                    feature,
                    options=options
                )

        selected_model = st.selectbox(
            "Pilih model prediksi",
            ["Logistic Regression", "Decision Tree"]
        )

        submitted = st.form_submit_button("Prediksi Kategori Sleep Debt")

    if submitted:
        # Mengubah input pengguna menjadi DataFrame dengan urutan kolom tepat.
        input_df = pd.DataFrame([input_values], columns=feature_cols)

        if selected_model == "Logistic Regression":
            selected_pipeline = st.session_state["logistic_model"]
        else:
            selected_pipeline = st.session_state["decision_tree_model"]

        # Melakukan prediksi kategori dan probabilitas kelas.
        predicted_class = selected_pipeline.predict(input_df)[0]
        probabilities = selected_pipeline.predict_proba(input_df)[0]
        class_names = selected_pipeline.named_steps["classifier"].classes_

        st.success(f"Prediksi kategori: *{predicted_class}*")

        probability_df = pd.DataFrame({
            "Kategori": class_names,
            "Probabilitas": probabilities
        })

        fig_prob = px.bar(
            probability_df,
            x="Kategori",
            y="Probabilitas",
            color="Kategori",
            title="Probabilitas Prediksi Setiap Kategori"
        )
        fig_prob.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(fig_prob, use_container_width=True)

# catatan dan batasan
st.markdown("---")
st.caption(
    "Catatan: Dashboard ini merupakan alat pembelajaran dan eksplorasi "
    "Machine Learning, bukan alat diagnosis medis. Hasil prediksi menunjukkan "
    "pola dalam dataset dan tidak membuktikan hubungan sebab-akibat."
)
