import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from sklearn.preprocessing import StandardScaler

# FUNGSI UNTUK MEMUAT DATA SQLITE 
@st.cache_data
def load_sql_data(db_path):
    """
    Memuat data dari database SQLite.
    Akan menangani kesalahan jika file tidak ditemukan.
    """
    try:
        # PENTING: Mengubah path file agar sesuai dengan struktur direktori Anda
        db_path = "Data/database.sqlite"
        conn = sqlite3.connect(db_path)
        player_df = pd.read_sql_query("SELECT * FROM Player", conn)
        player_attr_df = pd.read_sql_query("SELECT * FROM Player_Attributes", conn)
        conn.close()
        return player_df, player_attr_df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat database SQLite: {e}. Pastikan file 'database.sqlite' ada di direktori yang sama.")
        return None, None

# SISTEM REKOMENDASI

st.title("Sistem Rekomendasi Pemain Bola")
st.write("Halaman ini menggunakan sistem rekomendasi Content-Based Filtering untuk menemukan pemain serupa berdasarkan atribut teknis mereka.")
st.write("---")

# Muat data pemain dari database SQLite
player_df, player_attr_df = load_sql_data("database.sqlite")

if player_df is not None and player_attr_df is not None:
    # PRA-PEMROSESAN DATA
    st.info("Memproses data pemain...")

    # Gabungkan data pemain dan atribut, pilih atribut terbaru
    latest_attr_df = player_attr_df.sort_values('date').drop_duplicates('player_api_id', keep='last')
    combined_df = pd.merge(player_df, latest_attr_df, on='player_api_id', how='inner')

    # Membuat kolom 'passing' baru yang hilang
    combined_df['passing'] = (combined_df['short_passing'] + combined_df['long_passing']) / 2

    # fitur numerik yang akan digunakan untuk rekomendasi
    numeric_features = [
        'overall_rating', 'potential', 'crossing', 'finishing', 'heading_accuracy',
        'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
        'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
        'agility', 'reactions', 'balance', 'shot_power', 'jumping',
        'stamina', 'strength', 'long_shots', 'aggression', 'interceptions',
        'positioning', 'passing', 'vision', 'penalties', 'marking',
        'standing_tackle', 'sliding_tackle'
    ]
    
    # Tangani missing values dengan median
    for col in numeric_features:
        combined_df[col].fillna(combined_df[col].median(), inplace=True)
        
    # Normalisasi data numerik
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(combined_df[numeric_features])
    scaled_features_df = pd.DataFrame(scaled_features, columns=numeric_features)
    
    # Membuat vektor fitur
    features_for_recommender = scaled_features_df.copy()
    
    # Menambahkan data non-numerik untuk pencarian
    combined_df['full_name_and_id'] = combined_df['player_name'] + ' - ' + combined_df['player_api_id'].astype(str)
    
    st.success("Data pemain berhasil dimuat dan diproses!")

    # SISTEM REKOMENDASI INTERAKTIF
    st.subheader("Pilih Pemain Acuan")
    
    # Pilihan pemain
    player_names = combined_df['full_name_and_id'].tolist()
    # Mengubah st.selectbox untuk memungkinkan pencarian
    player_to_compare = st.selectbox(
        "Pilih pemain yang ingin Anda cari kesamaannya (Anda bisa mengetik nama di sini):",
        options=player_names
    )

    num_recommendations = st.slider(
        "Jumlah pemain yang direkomendasikan:",
        min_value=1, max_value=20, value=5
    )

    if st.button("Cari Rekomendasi"):
        if player_to_compare:
            # Dapatkan indeks pemain yang dipilih
            player_id_to_compare = int(player_to_compare.split(' - ')[-1])
            player_idx = combined_df[combined_df['player_api_id'] == player_id_to_compare].index[0]
            
            # Hitung Cosine Similarity
            cosine_sim = cosine_similarity(features_for_recommender)
            sim_scores = list(enumerate(cosine_sim[player_idx]))
            
            # Urutkan berdasarkan skor kesamaan
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Dapatkan top N rekomendasi (selain pemain itu sendiri)
            sim_scores = sim_scores[1:num_recommendations + 1]
            player_indices = [i[0] for i in sim_scores]
            
            recommended_players = combined_df.iloc[player_indices]
            
            st.subheader(f"Pemain yang direkomendasikan untuk {player_to_compare.split(' - ')[0]}:")
            
            # Tampilkan hasil
            display_df = recommended_players[['player_name', 'overall_rating', 'potential', 'preferred_foot']].copy()
            display_df.rename(columns={
                'player_name': 'Nama Pemain',
                'overall_rating': 'Rating Keseluruhan',
                'potential': 'Potensi',
                'preferred_foot': 'Kaki Utama'
            }, inplace=True)
            st.dataframe(display_df)