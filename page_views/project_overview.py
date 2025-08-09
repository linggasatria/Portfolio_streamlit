import streamlit as st
import pandas as pd
import numpy as np

# JUDUL HALAMAN
st.title("Proyek Analisis Data")
st.write("Jelajahi beberapa proyek analisis data yang telah saya kerjakan. Pilih salah satu proyek dari menu di bawah ini untuk melihat detailnya.")
st.write("---")

#CUSTOM CSS
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: #262730; /* Warna latar belakang gelap */
        border-radius: 10px; /* Sudut membulat */
        padding: 20px; /* Padding di dalam kotak */
        border: 1px solid #3d3e47; /* Garis tepi tipis */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Bayangan lembut */
    }
    div[data-testid="stMetricLabel"] {
        color: #e0e0e0; /* Warna teks label */
    }
    div[data-testid="stMetricValue"] {
        color: #f3f3f3; /* Warna teks nilai metrik */
        font-size: 3rem; /* Ukuran font lebih besar */
    }
    </style>
    """,
    unsafe_allow_html=True
)

#DATA PROYEK
project_selection = st.radio(
    "Pilih Proyek yang Ingin Anda Tampilkan:",
    ("California Housing", "Telco Customer Churn", "Titanic Survival Prediction")
)
st.write("---")

# FUNGSI LOAD DATASET
@st.cache_data
def load_data(file_path):
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
        st.error(f"Error: File '{file_path}' tidak ditemukan. Mohon pastikan file ada di direktori yang benar.")
        return None

# MENAMPILKAN KONTEN BERDASARKAN PILIHAN

# PROYEK: CALIFORNIA HOUSING

if project_selection == "California Housing":
    st.header("California Housing Price Prediction")
    
    st.subheader("Project Description")
    st.write(
        """
        Proyek ini merupakan analisis prediktif untuk memodelkan dan memprediksi harga 
        rumah di berbagai distrik di California. Dengan menggunakan teknik machine learning, 
        kita dapat mengidentifikasi variabel-variabel kunci yang memengaruhi nilai 
        properti dan membuat prediksi yang akurat.
        """
    )
    
# Dataset Overview
    st.subheader("Dataset Overview")
    st.write("Ringkasan statistik kunci dari dataset California Housing.")
    
    df = load_data('./Data/california_dataset.csv')
    if df is not None:
        total_rows = len(df)
        total_features = len(df.columns)
        avg_house_age = df['HouseAge'].mean()
        avg_income = df['MedInc'].mean() * 10000
        avg_house_price = df['house_price'].mean() * 100000
    

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sampel", f"{total_rows:,}")
        with col2:
            st.metric("Jumlah Fitur", total_features)
        with col3:
            st.metric("Rata-rata Usia Rumah", f"{avg_house_age:.1f} th")
    
        st.write("") 
        
        col4, col5 = st.columns(2)
        with col4:
            st.metric("Rata-rata Pendapatan", f"${avg_income:.2f}")
        with col5:
            st.metric("Rata-rata Harga Rumah", f"${avg_house_price:,.2f}")
    
    
#Feature Definition
    st.subheader("Feature Definition")
    st.markdown("""
    | Fitur | Deskripsi |
    | :--- | :--- |
    | **MedInc** | Median income in block group |
    | **HouseAge** | Median house age in block group |
    | **AveRooms** | Average number of rooms per household |
    | **AveBedrms** | Average number of bedrooms per household |
    | **Population** | Block group population |
    | **AveOccup** | Average household occupancy |
    | **Latitude** | Lokasi geografis (lintang) |
    | **Longitude** | Lokasi geografis (bujur) |
    | **house_price** | Harga rumah (target) |
    """)

#Business Context
    st.subheader("Business Context")
    st.write(
        """
        Model prediksi harga rumah sangat penting bagi industri real estate. 
        Model ini dapat membantu agen properti menentukan harga jual yang kompetitif, 
        membantu pembeli dalam estimasi nilai properti, dan juga dapat digunakan 
        oleh bank untuk evaluasi pinjaman properti.
        """
    )


# PROYEK: TELCO CUSTOMER CHURN

elif project_selection == "Telco Customer Churn":
    st.header("Telco Customer Churn Prediction")
    
    st.subheader("Project Description")
    st.write(
        """
        Proyek ini bertujuan untuk membangun model yang dapat memprediksi pelanggan 
        yang berpotensi untuk berhenti berlangganan (churn) dari layanan telekomunikasi. 
        Dengan mengidentifikasi pelanggan berisiko, perusahaan dapat mengambil tindakan 
        proaktif untuk meningkatkan retensi.
        """
    )

# Dataset Overview
    st.subheader("Dataset Overview")
    st.write("Ringkasan statistik kunci dari dataset Telco Customer Churn.")

    df = load_data('./Data/Telco_customer_churn.csv')
    if df is not None:

        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
        
        total_customers = len(df)
        churn_rate = (df['Churn Label'].value_counts(normalize=True).get('Yes', 0) * 100)
        features_count = len(df.columns)
        avg_tenure = df['Tenure Months'].mean()
        avg_monthly_charges = df['Monthly Charges'].mean()
        avg_total_charges = df['Total Charges'].mean()
        retention_rate = 100 - churn_rate
    

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", f"{total_customers:,}")
        with col2:
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        with col3:
            st.metric("Features", features_count)
        with col4:
            st.metric("Avg Tenure (months)", f"{avg_tenure:.1f}")
    
        col5, col6, col7 = st.columns(3)
        with col5:
            st.metric("Avg Monthly Charges", f"${avg_monthly_charges:.2f}")
        with col6:
            st.metric("Avg Total Charges", f"${avg_total_charges:,.2f}")
        with col7:
            st.metric("Retention Rate", f"{retention_rate:.1f}%")

# Feature Definition
    st.subheader("Feature Definition")
    st.markdown("""
    | Fitur | Deskripsi |
    | :--- | :--- |
    | **customerID** | ID unik untuk setiap pelanggan |
    | **gender** | Jenis kelamin pelanggan |
    | **tenure** | Durasi pelanggan menggunakan layanan (dalam bulan) |
    | **PhoneService** | Apakah pelanggan memiliki layanan telepon |
    | **InternetService** | Jenis layanan internet (DSL, Fiber optic, No) |
    | **Churn** | Apakah pelanggan berhenti berlangganan (Yes/No) |
    """)

#Business Context
    st.subheader("Business Context")
    st.write(
        """
        Retensi pelanggan adalah salah satu metrik terpenting dalam industri telekomunikasi. 
        Dengan model prediksi churn, perusahaan dapat mempersonalisasi penawaran atau 
        layanan khusus kepada pelanggan yang berisiko churn, sehingga dapat 
        mengurangi tingkat churn dan meningkatkan profitabilitas.
        """
    )

# 3. PROYEK: TITANIC SURVIVAL

elif project_selection == "Titanic Survival Prediction":
    st.header("Titanic Survival Prediction")
    
    st.subheader("Project Description")
    st.write(
        """
        Ini adalah proyek klasifikasi biner yang mencoba memprediksi apakah seorang 
        penumpang di kapal Titanic akan selamat atau tidak. Analisis ini mengeksplorasi 
        hubungan antara kelangsungan hidup dengan faktor-faktor seperti usia, kelas 
        penumpang, dan jenis kelamin.
        """
    )

# Dataset Overview (Modifikasi)
    st.subheader("Dataset Overview")
    st.write("Ringkasan statistik kunci dari dataset Titanic.")

    df = load_data('./Data/titanic.xlsx')
    if df is not None:
        total_passengers = len(df)
        survival_rate = (df['survived'].mean() * 100)
        features_count = len(df.columns)
        avg_age = df['age'].mean()
        female_passengers = len(df[df['sex'] == 'female'])
        male_passengers = len(df[df['sex'] == 'male'])
    
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Penumpang", f"{total_passengers:,}")
        with col2:
            st.metric("Tingkat Kelangsungan Hidup", f"{survival_rate:.1f}%")
        with col3:
            st.metric("Jumlah Fitur", features_count)
        
        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Penumpang Wanita", f"{female_passengers:,}")
        with col5:
            st.metric("Penumpang Pria", f"{male_passengers:,}")
        with col6:
            st.metric("Rata-rata Usia", f"{avg_age:.1f} th")

#Feature Definition
    st.subheader("Feature Definition")
    st.markdown("""
    | Fitur | Deskripsi |
    | :--- | :--- |
    | **PassengerId** | ID unik penumpang |
    | **Survived** | Kelangsungan hidup (0 = Tidak, 1 = Ya) |
    | **Pclass** | Kelas tiket (1 = Atas, 2 = Tengah, 3 = Bawah) |
    | **Name** | Nama penumpang |
    | **Sex** | Jenis kelamin |
    | **Age** | Usia penumpang |
    """)

 # Business Context
    st.subheader("Business Context")
    st.write(
        """
        Walaupun berdasarkan peristiwa bersejarah, proyek ini adalah contoh yang 
        sangat baik untuk memulai dalam data science, terutama untuk pembelajaran 
        model klasifikasi, pembersihan data (data cleaning), dan rekayasa fitur (feature engineering).
        """
    )