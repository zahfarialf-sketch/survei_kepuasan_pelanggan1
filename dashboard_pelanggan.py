import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(page_title="Dashboard Kepuasan Pelanggan", layout="wide")

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_ Harris=True)

st.title("🛍️ Dashboard Analisis Kepuasan Pelanggan")
st.markdown("Analisis data survei untuk optimasi layanan dan strategi segmentasi.")

# ==========================================================
# LOAD DATA
# ==========================================================
# Ganti nama file jika berbeda
FILE_NAME = "Data_Survei_Pelanggan_Toko.xlsx"

if os.path.exists(FILE_NAME):
    df = pd.read_excel(FILE_NAME)
    
    # Identifikasi kolom secara otomatis
    # Kolom 0: Nama, Kolom 1-4: Indikator (V1-V4), Kolom 5: Keseluruhan (V5)
    indikator = df.columns[1:5]  # Mengambil Kecepatan, Kualitas, Chat, Aplikasi
    target = df.columns[5]       # Mengambil Kepuasan Keseluruhan
    
    # ==========================================================
    # HEADER KPI
    # ==========================================================
    avg_total = df[target].mean()
    ikm_val = (avg_total / 5) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rata-rata Skor", f"{avg_total:.2f} / 5.0")
    col2.metric("Indeks Kepuasan", f"{ikm_val:.1f}%")
    col3.metric("Total Responden", len(df))
    
    status = "Sangat Baik" if ikm_val >= 80 else "Baik" if ikm_val >= 60 else "Cukup"
    col4.metric("Status Layanan", status)

    st.divider()

    # ==========================================================
    # BAGIAN TAB ANALISIS
    # ==========================================================
    tab1, tab2, tab3 = st.tabs(["📊 Analisis Performa", "🎯 Segmentasi Pelanggan", "🔍 Detail Data"])

    with tab1:
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.subheader("Skor Rata-rata per Indikator")
            fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
            mean_vals = df[indikator].mean()
            sns.barplot(x=mean_vals.values, y=mean_vals.index, palette="Blues_d", ax=ax_bar)
            ax_bar.set_xlim(0, 5)
            # Garis bantu target
            ax_bar.axvline(4.0, color='red', linestyle='--', label='Target (4.0)')
            st.pyplot(fig_bar)
        
        with c2:
            st.subheader("Distribusi Kepuasan")
            fig_dist, ax_dist = plt.subplots(figsize=(8, 6))
            sns.histplot(df[target], bins=5, kde=True, color="skyblue", ax=ax_dist)
            ax_dist.set_title("Penyebaran Skor Keseluruhan")
            st.pyplot(fig_dist)

    with tab2:
        st.subheader("Pengelompokan Pelanggan (Clustering)")
        
        # Proses K-Means
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[indikator])
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Mapping nama cluster berdasarkan rata-rata skor
        cluster_means = df.groupby('Cluster')[target].mean().sort_values(ascending=False)
        mapping = {
            cluster_means.index[0]: "Loyalist (Puas)",
            cluster_means.index[1]: "Neutral (Biasa Saja)",
            cluster_means.index[2]: "At Risk (Tidak Puas)"
        }
        df['Segmen'] = df['Cluster'].map(mapping)
        
        c1, c2 = st.columns([1, 1.2])
        
        with c1:
            st.write("Proporsi Segmen Pelanggan")
            fig_pie, ax_pie = plt.subplots()
            df['Segmen'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette("pastel"), ax=ax_pie)
            ax_pie.set_ylabel("")
            st.pyplot(fig_pie)
            
        with c2:
            st.write("Karakteristik Tiap Segmen")
            summary = df.groupby('Segmen')[indikator].mean()
            st.dataframe(summary.style.background_gradient(cmap='YlGn'))

    with tab3:
        st.subheader("Tabel Data Survei")
        st.dataframe(df, use_container_width=True)
        
        # Tombol Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Hasil Analisis (CSV)", csv, "hasil_analisis.csv", "text/csv")

else:
    st.error(f"⚠️ File **{FILE_NAME}** tidak ditemukan!")
    st.info("Pastikan Anda sudah mengunggah file Excel dengan nama yang tepat ke GitHub.")

