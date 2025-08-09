import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


# FUNGSI UNTUK MEMUAT DAN MEMPROSES DATA
@st.cache_resource
def load_and_preprocess_data(uploaded_file, target_column, problem_type):
    """
    Memuat data, melakukan pra-pemrosesan generik, dan membagi data.
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Format file tidak didukung. Mohon unggah file .csv atau .xlsx.")
            return None, None, None, None, None

        if target_column not in df.columns:
            st.error(f"Kolom target '{target_column}' tidak ditemukan di dataset.")
            return None, None, None, None, None

        # PRA-PEMROSESAN OTOMATIS
        # Pisahkan fitur dan target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Inisialisasi dictionary untuk menyimpan LabelEncoder
        label_encoders = {}
        
        # Tangani nilai yang hilang dan encode kolom kategorikal
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                # Tangani NaN sebelum encoding
                X[col].fillna('missing', inplace=True)
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le
            else:
                # Isi nilai numerik yang hilang dengan median
                X[col].fillna(X[col].median(), inplace=True)
        
        # Tangani nilai yang hilang di kolom target (jika ada)
        if problem_type == 'Klasifikasi':
            if y.dtype == 'object':
                le = LabelEncoder()
                y.fillna(y.mode()[0], inplace=True)
                y = le.fit_transform(y)
                label_encoders[target_column] = le
        elif problem_type == 'Regresi':
            if y.dtype in ['int64', 'float64']:
                y.fillna(y.median(), inplace=True)
            
        return X, y, df, label_encoders, y.dtype

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat atau memproses data: {e}")
        return None, None, None, None, None

# FUNGSI UNTUK MELATIH MODEL
@st.cache_resource
def train_model(X, y, problem_type):
    """
    Melatih model yang sesuai berdasarkan jenis masalah.
    """
    if problem_type == 'Klasifikasi':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif problem_type == 'Regresi':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X, y)
    return model

#  FUNGSI UNTUK MENAMPILKAN METRIK
def display_metrics(y_true, y_pred, problem_type):
    """
    Menampilkan metrik yang relevan berdasarkan jenis masalah.
    """
    if problem_type == 'Klasifikasi':
        st.subheader("Metrik Performa (Klasifikasi)")
        accuracy = accuracy_score(y_true, y_pred)
        st.metric("Akurasi", f"{accuracy:.2f}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Prediksi')
        ax.set_ylabel('Aktual')
        st.pyplot(fig)
    
    elif problem_type == 'Regresi':
        st.subheader("Metrik Performa (Regresi)")
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
        with col2:
            st.metric("R-squared", f"{r2:.2f}")

# APLIKASI STREAMLIT UTAMA
st.title("Pipeline Machine Learning Generik dengan Streamlit")
st.write("Unggah dataset Anda dan tentukan jenis masalah untuk melatih model.")
st.write("---")

# Input pengguna
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("1. Unggah Dataset (CSV/Excel)", type=["csv", "xlsx"])
with col2:
    problem_type = st.radio("2. Pilih Jenis Masalah:", ("Klasifikasi", "Regresi"))

if uploaded_file:
    # Menggunakan placeholder agar pengguna bisa memasukkan nama kolom yang benar
    target_column = st.text_input("3. Masukkan Nama Kolom Target:", placeholder="Contoh: Survived, Churn, MedHouseVal, house_price")

    if target_column:
        X, y, df, label_encoders, y_dtype = load_and_preprocess_data(uploaded_file, target_column, problem_type)

        if X is not None and y is not None:
            # Latih model
            with st.spinner('Melatih model...'):
                model = train_model(X, y, problem_type)
            st.success('Model berhasil dilatih!')
            
            # Tampilkan metrik performa pada data pelatihan
            y_pred = model.predict(X)
            display_metrics(y, y_pred, problem_type)

            st.write("---")
            st.subheader("Prediksi Data Baru")
            st.write("Unggah file CSV/Excel lain untuk mendapatkan prediksi dari model yang telah dilatih.")

            uploaded_file_new = st.file_uploader("Unggah Dataset Baru untuk Prediksi", type=["csv", "xlsx"])

            if uploaded_file_new:
                try:
                    if uploaded_file_new.name.endswith('.csv'):
                        df_new = pd.read_csv(uploaded_file_new)
                    else:
                        df_new = pd.read_excel(uploaded_file_new)

                    st.dataframe(df_new.head())

                    if st.button("Lakukan Prediksi pada Data Baru"):
                        # Pra-proses data input baru
                        for col in X.columns:
                            if col in df_new.columns:
                                if X[col].dtype == 'object' and col in label_encoders:
                                    df_new[col].fillna('missing', inplace=True)
                                    df_new[col] = label_encoders[col].transform(df_new[col])
                                elif X[col].dtype in ['int64', 'float64']:
                                    df_new[col].fillna(df_new[col].median(), inplace=True)
                            else:
                                st.warning(f"Kolom '{col}' tidak ditemukan di dataset baru. Prediksi tidak dapat dilakukan.")
                                st.stop()

                        # Hapus kolom target jika ada di data baru
                        if target_column in df_new.columns:
                            df_new_features = df_new.drop(columns=[target_column])
                        else:
                            df_new_features = df_new

                        # Pastikan kolom sesuai
                        if not set(X.columns).issubset(set(df_new_features.columns)):
                            st.error("Kolom pada dataset baru tidak sesuai dengan kolom yang digunakan untuk melatih model.")
                        else:
                            predictions = model.predict(df_new_features[X.columns])
                            
                            df_new['Prediksi'] = predictions
                            
                            if problem_type == 'Klasifikasi' and target_column in label_encoders:
                                df_new['Prediksi'] = label_encoders[target_column].inverse_transform(df_new['Prediksi'].astype(int))

                            st.subheader("Hasil Prediksi")
                            st.dataframe(df_new)
                            st.success("Prediksi berhasil dilakukan!")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses file baru: {e}")
