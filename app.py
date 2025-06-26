import streamlit as st
import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import threading
import time

# Konfigurasi folder history
HISTORY_DIR = "simulasi_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

# --- Navigation Bar ---
st.markdown("""
<style>
    .navbar {
        background-color: #0078D4;
        padding: 10px;
        font-family: sans-serif;
        color: white;
        display: flex;
        justify-content: start;
        gap: 30px;
        font-weight: bold;
        font-size: 16px;
    }
    .navbar a {
        color: white;
        text-decoration: none;
        cursor: pointer;
        padding: 5px 10px;
        border-radius: 4px;
        transition: background-color 0.3s ease;
    }
    .navbar a:hover {
        background-color: #005fa3;
    }
</style>
<div class="navbar">
    <a href="?tab=Home" target="_self">Home</a>
    <a href="?tab=History" target="_self">History</a>
</div>
""", unsafe_allow_html=True)

# --- Tab Manual (gunakan query params) ---
query_params = st.experimental_get_query_params()
current_tab = query_params.get("tab", ["Home"])[0]

if current_tab == "Home":
    st.title("Dashboard Simulasi VANET dengan RL")

    with st.form(key='simulasi_form'):
        uploaded_file = st.file_uploader("Upload FCD File (.csv/.xml)", type=["csv", "xml"])

        enable_rl = st.checkbox("Enable Reinforcement Learning?")

        # Selalu tampilkan dropdown algoritma
        algoritma = st.selectbox("Pilih Algoritma Optimasi", ["Q-learning", "SAC"])

        # Selalu tampilkan mode training/testing
        mode_latih_uji = st.radio("Mode", ("Training", "Testing"))

        submit_button = st.form_submit_button(label='Run Simulation')

    if submit_button:
        if uploaded_file is None:
            st.error("Silakan upload file FCD sebelum menjalankan simulasi.")
        else:
            st.info("Memulai simulasi...")

            # Hapus file lama jika ada
            FCD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fcd-input.xml")
            if os.path.exists(FCD_PATH):
                os.remove(FCD_PATH)

            # Simpan file FCD langsung ke folder simulasi_vanet.py
            with open(FCD_PATH, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Fungsi untuk menjalankan algoritma
            def run_algorithm():
                if enable_rl:
                    if mode_latih_uji == "Training":
                        st.info(f"Menjalankan {algoritma} - Mode Training...")
                        if algoritma == "Q-learning":
                            subprocess.run(["python", "-m", "Q-learning.train"])
                        else:
                            subprocess.run(["python", "-m", "SAC.train"])
                    else:
                        st.warning("Mode Testing belum diimplementasikan.")
                else:
                    st.info("Menjalankan baseline...")
                    subprocess.run(["python", "-m", "baseline.run"])

            # Fungsi untuk delay dan jalankan simulasi_vanet.py
            def delayed_simulation():
                time.sleep(10)  # Delay 10 detik
                st.info("Menjalankan simulasi VANET...")
                subprocess.run(["python", "simulasi_vanet.py"])

            # Jalankan kedua proses secara bersamaan
            thread1 = threading.Thread(target=run_algorithm)
            thread2 = threading.Thread(target=delayed_simulation)

            thread1.start()
            thread2.start()

            thread1.join()
            thread2.join()

            st.success("Simulasi selesai!")

            # Buat direktori history dengan penomoran berurut
            existing_folders = [f for f in os.listdir(HISTORY_DIR) if os.path.isdir(os.path.join(HISTORY_DIR, f))]
            next_number = len(existing_folders) + 1
            sim_dir = os.path.join(HISTORY_DIR, str(next_number))
            os.makedirs(sim_dir)

            # Pindahkan file excel hasil simulasi ke folder history
            output_file = "output_simulasi.xlsx"
            if os.path.exists(output_file):
                os.rename(output_file, os.path.join(sim_dir, output_file))

            # Simpan konfigurasi
            config = {
                "file_fcd": uploaded_file.name,
                "enable_rl": enable_rl,
                "algoritma": algoritma,
                "mode": mode_latih_uji
            }

            with open(os.path.join(sim_dir, "config.txt"), "w") as f:
                for key, val in config.items():
                    f.write(f"{key}: {val}\n")

            st.success("Selesai. Hasil disimpan dalam history.")

elif current_tab == "History":
    st.title("History Simulasi")

    if not os.listdir(HISTORY_DIR):
        st.info("Belum ada data simulasi.")
    else:
        history_list = sorted([f for f in os.listdir(HISTORY_DIR) if os.path.isdir(os.path.join(HISTORY_DIR, f))], key=int)
        selected = st.selectbox("Pilih simulasi dari history:", history_list)

        sim_path = os.path.join(HISTORY_DIR, selected)
        config_path = os.path.join(sim_path, "config.txt")
        excel_path = os.path.join(sim_path, "output_simulasi.xlsx")

        with open(config_path, "r") as f:
            st.markdown("### Konfigurasi Simulasi:")
            st.code(f.read())

        if os.path.exists(excel_path):
            # Tombol Download Excel
            with open(excel_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download File Excel",
                    data=f,
                    file_name="output_simulasi.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            st.markdown("### Visualisasi Hasil Simulasi:")

            try:
                # Plot CBR
                df_ts = pd.read_excel(excel_path, sheet_name="Time_Series_Analysis").set_index('Timestamp')
                fig1, ax1 = plt.subplots()
                df_ts['CBR_mean'].plot(ax=ax1, title="CBR Mean per Timestamp", xlabel="Waktu", ylabel="CBR")
                st.pyplot(fig1)

                # Plot SINR
                fig2, ax2 = plt.subplots()
                df_ts['SINR_mean'].plot(ax=ax2, title="SINR Mean per Timestamp", xlabel="Waktu", ylabel="SINR")
                st.pyplot(fig2)

                # Plot Latency
                df_dr = pd.read_excel(excel_path, sheet_name="Detailed_Results")
                df_latency = df_dr.groupby('Timestamp')['Latency'].mean()
                fig3, ax3 = plt.subplots()
                df_latency.plot(ax=ax3, title="Rata-rata Latency per Timestamp", xlabel="Waktu", ylabel="Latency (ms)")
                st.pyplot(fig3)

                # Plot PDR (dikalikan dengan 100 untuk persentase)
                df_pdr = df_dr.groupby('Timestamp')['PDR'].mean() * 100
                fig4, ax4 = plt.subplots()
                df_pdr.plot(ax=ax4, title="Rata-rata PDR per Timestamp", xlabel="Waktu", ylabel="PDR (%)")
                st.pyplot(fig4)

            except Exception as e:
                st.error(f"Gagal membaca file Excel: {e}")
        else:
            st.warning("File hasil simulasi tidak ditemukan.")