import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import joblib

# Set Seaborn style
sns.set(style="whitegrid")

# Judul Dashboard
st.set_page_config(page_title="Dashboard Performa Siswa 🎓", page_icon="📚", layout="wide")
st.title("📊 Dashboard Performa Siswa 🎓")
st.write("Selamat datang di dashboard interaktif yang penuh warna dan seru! 🌈")

# Sidebar untuk navigasi
st.sidebar.title("Navigasi 🧭")
options = st.sidebar.radio("Pilih Halaman:", ["Home 🏠", "Data 📑", "Statistik Deskriptif 📊", "Visualisasi 📈", "Analisis Lanjutan 🔍", "Prediksi Nilai Siswa 🔮"])

# Mengunggah Dataset
@st.cache_data
def load_data():
    data = pd.read_csv('StudentPerformanceFactors2.csv')
    return data

df = load_data()

# Halaman Home
if options == "Home 🏠":
    st.subheader("Selamat datang di Dashboard Performa Siswa!")
    st.write("Gunakan menu di sisi kiri untuk menjelajahi data dan analisis.")

# Halaman Data
elif options == "Data 📑":
    st.subheader("📋 Dataset")
    st.write(df.head())
    if st.checkbox("Tampilkan Seluruh Data"):
        st.write(df)

# Halaman Statistik Deskriptif
elif options == "Statistik Deskriptif 📊":
    st.subheader("📊 Statistik Deskriptif")
    st.write(df.describe())
    # Menambahkan statistik tambahan
    st.metric("Total Siswa", df.shape[0])
    st.metric("Rata-rata Nilai Ujian", f"{df['Nilai_Ujian'].mean():.2f}")
    st.metric("Jam Belajar Maksimal", f"{df['Jam_Belajar'].max()} jam")

# Halaman Visualisasi
elif options == "Visualisasi 📈":
    st.subheader("Visualisasi Korelasi 🔗")

    # Visualisasi Korelasi dengan Heatmap
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(
        df.corr(),
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Heatmap Korelasi", fontsize=18)
    st.pyplot(fig)

    # Scatter Plot Interaktif dengan Seaborn dan Plotly
    st.subheader("📍 Pengaruh Faktor Terhadap Nilai Ujian")
    x_axis = st.selectbox("Pilih Faktor X:", df.columns.drop("Nilai_Ujian"))
    fig = px.scatter(df, x=x_axis, y="Nilai_Ujian", color="Jenis_Kelamin", title=f"{x_axis} vs Nilai Ujian",
                     labels={x_axis: x_axis, "Nilai_Ujian": "Nilai Ujian"}, color_discrete_sequence=["#FF6F61", "#6B5B95"])
    st.plotly_chart(fig, use_container_width=True)

    # Histogram Jam Belajar
    st.subheader("⏰ Distribusi Jam Belajar")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["Jam_Belajar"], kde=True, color="skyblue", ax=ax)
    plt.xlabel("Jam Belajar")
    plt.ylabel("Frekuensi")
    plt.title("Distribusi Jam Belajar", fontsize=16)
    st.pyplot(fig)

# Halaman Analisis Lanjutan
elif options == "Analisis Lanjutan 🔍":
    st.subheader("💡 Rata-rata Nilai Ujian Berdasarkan Faktor")
    faktor = st.selectbox("Pilih Faktor Kategori:", [
        "Skor_Sebelumnya", "Kehadiran", "Keterlibatan_OrangTua",
        "Akses_ke_Sumber_Daya", "Kegiatan_Ekstrakurikuler", "Motivasi_Level",
        "Akses_Internet", "Pendapatan_Keluarga", "Kualitas_Guru",
        "Jenis_Sekolah", "Pengaruh_Teman_Sebaya", "Gangguan_Belajar",
        "Tingkat_Pendidikan_OrangTua", "Jarak_Dari_Rumah", "Jenis_Kelamin"
    ])
    mean_scores = df.groupby(faktor)["Nilai_Ujian"].mean()
    st.bar_chart(mean_scores)

    # Pie Chart untuk Distribusi Jenis Kelamin
    st.subheader("🍰 Distribusi Jenis Kelamin")
    gender_counts = df["Jenis_Kelamin"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    ax.axis('equal')
    st.pyplot(fig)

    # Menambahkan bagian Tips Acak
    st.subheader("💡 Tips Belajar")
    study_tips = [
        "Istirahat yang cukup! 💤",
        "Buat jadwal belajar yang teratur! 📅",
        "Cari teman belajar yang menyenangkan! 👫",
        "Jangan lupa olahraga ringan! 🏃",
        "Gunakan teknik membaca cepat! 📖",
        "Jangan lupa minum air saat belajar! 💧"
    ]
    if st.button("Dapatkan Tips Acak"):
        st.write(np.random.choice(study_tips))

# Halaman Prediksi Nilai Siswa
elif options == "Prediksi Nilai Siswa 🔮":
    # Memuat model yang telah disimpan
    model = joblib.load('prediksi_nilai_siswa.pkl')

    # Definisi mapping untuk kategori
    Peer_Influence = {
        'High': 0,
        'Medium': 1,
        'Low': 2,
    }

    Study_Disruptions = {
        'None': 0,
        'Mild': 1,
        'Severe': 2,
    }

    # Fungsi untuk melakukan prediksi
    def predict(input_data):
        # Konversi input menjadi DataFrame dengan format yang benar
        input_df = pd.DataFrame(input_data, index=[0])
        
        # Pastikan urutan kolom sesuai dengan model
        expected_columns = ['Jam_Belajar', 'Kehadiran', 'Sesi_Bimbingan_Belajar', 'Pengaruh_Teman_Sebaya', 'Gangguan_Belajar']
        input_df = input_df[expected_columns]
        
        return model.predict(input_df)

    st.subheader("🎓 Sistem Prediksi Nilai Siswa")

    # Kolom untuk input
    col1, col2 = st.columns(2)

    with col1:
        st.header("Input Data Siswa")
        
        # Input jam belajar
        Jam_Belajar = st.number_input(
            'Jam Belajar per minggu', 
            min_value=0.0, 
            value=3.0,
            step=0.1,
            help="Total jam belajar per minggu"
        )
        
        # Input kehadiran
        Kehadiran = st.number_input(
            'Kehadiran (%)', 
            min_value=0.0, 
            max_value=100.0, 
            value=75.0,
            step=0.1,
            help="Persentase kehadiran siswa di kelas"
        )

        # Input sesi bimbingan belajar
        Sesi_Bimbingan_Belajar = st.number_input(
            'Sesi Bimbingan Belajar per minggu', 
            min_value=0, 
            value=2,
            step=1,
            help="Jumlah sesi bimbingan belajar per minggu"
        )

    with col2:
        st.header("Faktor Tambahan")
        
        # Pilih pengaruh teman sebaya
        peer_influence = st.selectbox(
            "Pengaruh Teman Sebaya", 
            list(Peer_Influence.keys()),
            help="Tingkat pengaruh teman sebaya terhadap siswa"
        )
        
        # Pilih gangguan belajar
        study_disruptions = st.selectbox(
            "Gangguan Belajar", 
            list(Study_Disruptions.keys()),
            help="Tingkat gangguan belajar yang dialami siswa"
        )

    # Tombol prediksi
    predict_button = st.button("Prediksi Nilai")

    # Tampilkan hasil prediksi
    if predict_button:
        # Membuat dictionary input dengan format yang benar
        input_data = {
            'Jam_Belajar': Jam_Belajar,
            'Kehadiran': Kehadiran,
            'Sesi_Bimbingan_Belajar': Sesi_Bimbingan_Belajar,
            'Pengaruh_Teman_Sebaya': Peer_Influence[peer_influence],
            'Gangguan_Belajar': Study_Disruptions[study_disruptions]
        }
        
        try:
            # Lakukan prediksi
            predicted_values = predict(input_data)
            
            # Tampilkan hasil
            st.header("Hasil Prediksi")
            
            # Kolom untuk menampilkan informasi
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.metric(
                    label="Skor Prediksi", 
                    value=f"{predicted_values[0]:.2f}"
                )
            
            with result_col2:
                # Berikan interpretasi skor
                if predicted_values[0] < 50:
                    st.warning("🚨 Skor di bawah rata-rata. Perlu perbaikan.")
                elif 50 <= predicted_values[0] < 70:
                    st.info("⚠️ Skor cukup. Masih ada ruang untuk peningkatan.")
                else:
                    st.success("🎉 Skor bagus! Pertahankan prestasi.")
            
            # Visualisasi faktor input
            st.subheader("Faktor yang Dipertimbangkan")
            
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
            st.error("Pastikan model sudah dilatih dengan benar.")

# Footer
st.sidebar.subheader("Tentang")
st.sidebar.write("Dashboard ini dibuat untuk menganalisis data performa siswa dan memberikan tips belajar interaktif.")
st.sidebar.write("Dibuat Oleh Kelompok 5")
