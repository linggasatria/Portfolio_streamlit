import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# --- FUNGSI UNTUK MEMUAT DATASET ---
@st.cache_data
def load_data(file_path):
    """
    Memuat dataset dari file. Jika file tidak ditemukan, akan menampilkan peringatan.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            st.error(f"Error: Format file '{file_path}' tidak didukung. Mohon gunakan .csv atau .xlsx.")
            return None
        return df
    except FileNotFoundError:
        st.warning(f"File '{file_path}' tidak ditemukan. Mohon pastikan file ada di direktori yang benar.")
        return None

# ==================================
# KONTEN HALAMAN: VISUALISASI & PERFORMA
# ==================================
st.header("Visualisasi Data & Performa Model")
st.write("Halaman ini menampilkan visualisasi dataset dan performa model machine learning untuk proyek Titanic Survival Prediction.")
st.write("---")

# Muat dataset Titanic
# Menggunakan path file yang diperbarui
df_titanic = load_data('./Data/titanic.xlsx')
if df_titanic is None:
    st.stop() # Hentikan eksekusi jika data tidak ditemukan

# --- PRA-PEMROSESAN DATA (DATA PREPROCESSING) ---
# Mengisi nilai Age yang hilang dengan nilai median
df_titanic['age'].fillna(df_titanic['age'].median(), inplace=True)
# Mengisi nilai Fare yang hilang dengan nilai median
df_titanic['fare'].fillna(df_titanic['fare'].median(), inplace=True)
# Mengubah kolom kategorikal 'Sex' dan 'Embarked' menjadi numerik
df_titanic['sex'] = df_titanic['sex'].map({'male': 0, 'female': 1})
df_titanic['embarked'] = df_titanic['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
# Mengisi nilai Embarked yang hilang dengan mode (nilai yang paling sering muncul)
df_titanic['embarked'].fillna(df_titanic['embarked'].mode()[0], inplace=True)

# --- DEFINISIKAN FITUR (X) DAN TARGET (y) ---
# Kita akan menggunakan beberapa fitur untuk memprediksi kelangsungan hidup
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target = 'survived'
X = df_titanic[features]
y = df_titanic[target]

# --- MEMBAGI DATA MENJADI DATA LATIH DAN DATA UJI ---
# Memisahkan 80% data untuk pelatihan dan 20% untuk pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- VISUALISASI DATASET ---
st.subheader("Visualisasi Dataset")
st.write("Berikut adalah beberapa visualisasi untuk memahami dataset Titanic.")

# Visualisasi 1: Distribusi Usia
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df_titanic['age'].dropna(), bins=30, kde=True, ax=ax)
ax.set_title('Distribusi Usia Penumpang')
ax.set_xlabel('Usia')
ax.set_ylabel('Jumlah Penumpang')
st.pyplot(fig)

# Visualisasi 2: Tingkat Kelangsungan Hidup berdasarkan Jenis Kelamin
df_visual = df_titanic.copy()
# Konversi kembali kolom 'Sex' menjadi label kategorikal
df_visual['sex'] = df_visual['sex'].map({0: 'male', 1: 'female'})

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=df_visual['sex'], y=df_visual['survived'], ax=ax, palette='Set2')
ax.set_title('Tingkat Kelangsungan Hidup Berdasarkan Jenis Kelamin')
ax.set_xlabel('Jenis Kelamin')
ax.set_ylabel('Persentase Selamat')
st.pyplot(fig)
st.write("Terlihat bahwa tingkat kelangsungan hidup penumpang wanita jauh lebih tinggi.")

st.write("---")

# --- MEMBUAT DAN MELATIH MODEL ---
st.subheader("Performa Model Machine Learning")
st.write("Pilih model di bawah ini untuk melihat metrik performa dan confusion matrix-nya.")

# Inisialisasi dan latih model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

model_lr = LogisticRegression(random_state=42, max_iter=200)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# --- MENGHITUNG METRIK UNTUK MODEL ASLI ---
models = {
    "Random Forest Classifier": {
        "y_pred": y_pred_rf,
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "precision": precision_score(y_test, y_pred_rf),
        "recall": recall_score(y_test, y_pred_rf),
        "f1_score": f1_score(y_test, y_pred_rf)
    },
    "Logistic Regression": {
        "y_pred": y_pred_lr,
        "accuracy": accuracy_score(y_test, y_pred_lr),
        "precision": precision_score(y_test, y_pred_lr),
        "recall": recall_score(y_test, y_pred_lr),
        "f1_score": f1_score(y_test, y_pred_lr)
    }
}

# Interaktif untuk memilih model
model_choice = st.selectbox("Pilih Model:", list(models.keys()))

if model_choice:
    selected_model = models[model_choice]
    st.write(f"### Metrik Performa untuk Model: {model_choice}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Akurasi", f"{selected_model['accuracy']:.2f}")
    with col2:
        st.metric("Presisi", f"{selected_model['precision']:.2f}")
    with col3:
        st.metric("Recall", f"{selected_model['recall']:.2f}")
    with col4:
        st.metric("F1-Score", f"{selected_model['f1_score']:.2f}")

    # Confusion Matrix
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, selected_model['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix - {model_choice}')
    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Aktual')
    ax.xaxis.set_ticklabels(['Tidak Selamat', 'Selamat'])
    ax.yaxis.set_ticklabels(['Tidak Selamat', 'Selamat'])
    st.pyplot(fig)