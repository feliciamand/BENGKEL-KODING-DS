import streamlit as st
import pandas as pd
import joblib

# Load model
model_rf = joblib.load(r"D:\feliz navidad\TUGAS KULIAH\BENGKEL KODING\UAS BENGKOD DS\PROGRESS UAS BENGKOD DS\PROGRESS 5_DEPLOYMENT\model_rf.pkl")


# Judul aplikasi
st.title("Prediksi Tingkat Obesitas Berdasarkan Gaya Hidup")

# Input data pasien
st.markdown("### Input Data Pasien")

age = st.number_input("Usia (tahun)", value=21)
gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
height = st.number_input("Tinggi badan (m)", value=1.62)
weight = st.number_input("Berat badan (kg)", value=64.0)
calc = st.selectbox("Kebiasaan konsumsi alkohol (CALC)", ["no", "Sometimes", "Frequently", "Always"])
favc = st.selectbox("Apakah sering konsumsi makanan tinggi kalori? (FAVC)", ["no", "yes"])
fcvc = st.selectbox("Frekuensi konsumsi sayur, skala (1-3) (FCVC)", [1, 2, 3])
ncp = st.selectbox("Jumlah makan besar per hari (NCP)", [1, 2, 3, 4])
scc = st.selectbox("Apakah konsumsi suplemen? (SCC)", ["no", "yes"])
smoke = st.selectbox("Apakah merokok? (SMOKE)", ["no", "yes"])
ch2o = st.selectbox("Konsumsi air putih per hari (CH2O)", [1, 2, 3])
family_history = st.selectbox("Apakah memiliki riwayat keluarga overweight?", ["no", "yes"])
faf = st.selectbox("Aktivitas fisik (jam/minggu) (FAF)", [0, 1, 2, 3])
tue = st.selectbox("Waktu layar per hari (jam) (TUE)", [0, 1, 2, 3])
caec = st.selectbox("Frekuensi camilan (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Transportasi utama (MTRANS)", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

# Encoding sesuai training
binary_map = {"no": 0, "yes": 1}
gender_map = {"Laki-laki": 1.0, "Perempuan": 0.0}
calc_map = {"no": 0.0, "Sometimes": 0.33, "Frequently": 0.66, "Always": 1.0}
caec_map = {"no": 0.0, "Sometimes": 0.33, "Frequently": 0.66, "Always": 1.0}
mtrans_map = {"Public_Transportation": 0.75, "Walking": 0.5, "Automobile": 1.0, "Motorbike": 0.25, "Bike": 0.0}

# Data input (tanpa scaling)
# Lanjutkan input_data
input_data = {
    'Age': age,
    'Gender': gender_map[gender],
    'Height': height,
    'Weight': weight,
    'CALC': calc_map[calc],
    'FAVC': binary_map[favc],
    'FCVC': fcvc,
    'NCP': ncp,
    'SCC': binary_map[scc],
    'SMOKE': binary_map[smoke],
    'CH2O': ch2o,
    'family_history_with_overweight': binary_map[family_history],
    'FAF': faf,
    'TUE': tue,
    'CAEC': caec_map[caec],
    'MTRANS': mtrans_map[mtrans]
}


# Konversi ke DataFrame
df_input = pd.DataFrame([input_data])

# Tombol prediksi
if st.button("Prediksi"):
    pred = model_rf.predict(df_input)[0]

    label_map = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Overweight_Level_I",
        3: "Overweight_Level_II",
        4: "Obesity_Type_I",
        5: "Obesity_Type_II",
        6: "Obesity_Type_III"
    }

    st.success(f"Hasil Prediksi: {label_map[pred]}")


