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

# --- Tab Manual (gunakan anchor link) ---
current_tab = st.experimental_get_query_params().get("tab", ["Home"])[0]

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

            # st.success(f"File FCD berhasil disimpan di {FCD_PATH}")

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
                time.sleep(10)  # Delay 5 detik
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

            # Buat direktori history
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sim_dir = os.path.join(HISTORY_DIR, timestamp)
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

            # --- Pop-up Konfirmasi Visualisasi ---
            st.markdown("""
            <style>
                .modal {
                    position: fixed;
                    top: 0; left: 0;
                    width: 100%; height: 100%;
                    background-color: rgba(0,0,0,0.5);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 9999;
                }
                .modal-content {
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0px 0px 10px #000;
                    text-align: center;
                    text-color: white;
                    width: 400px;
                }
                .btn-modal {
                    margin: 10px;
                    padding: 10px 20px;
                    font-size: 16px;
                    cursor: pointer;
                    border: none;
                    border-radius: 4px;
                }
                .btn-ya { background-co
                lor: #4CAF50; color: white; }
                .btn-tidak { background-color: #f44336; color: white; }
            </style>
            <div id="modal" class="modal">
              <div class="modal-content">
                <h3>Apakah ingin plot hasil simulasi?</h3>
                <button class="btn-modal btn-ya" onclick="window.location.reload(); window.parent.postMessage({isStreamlitDialog: true, runPlot: true}, '*')">Ya</button>
                <button class="btn-modal btn-tidak" onclick="window.location.reload()">Tidak</button>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Cek apakah user klik Ya atau Tidak
            query_params = st.query_params()
            if query_params.get("runPlot", [False])[0]:
                st.markdown("<script>window.document.getElementById('modal').style.display='none'</script>", unsafe_allow_html=True)

                excel_path = os.path.join(sim_dir, output_file)
                try:
                    df_time_series = pd.read_excel(excel_path, sheet_name="Time_Series_Analysis")
                    fig1, ax1 = plt.subplots()
                    df_time_series.set_index('Timestamp')[['CBR_mean', 'SINR_mean']].plot(ax=ax1)
                    ax1.set_title("CBR Mean & SINR Mean per Timestamp")
                    st.pyplot(fig1)

                    df_detailed = pd.read_excel(excel_path, sheet_name="Detailed_Results")
                    df_grouped = df_detailed.groupby('Timestamp')[['Latency', 'PDR']].mean()
                    fig2, ax2 = plt.subplots()
                    df_grouped.plot(ax=ax2)
                    ax2.set_title("Rata-rata Latency & PDR per Timestamp")
                    st.pyplot(fig2)

                    # Simpan plot
                    plot1_path = os.path.join(sim_dir, "time_series.png")
                    plot2_path = os.path.join(sim_dir, "detailed_results.png")
                    fig1.savefig(plot1_path)
                    fig2.savefig(plot2_path)

                except Exception as e:
                    st.error(f"Gagal membaca file Excel: {e}")

            st.success("Selesai. Hasil disimpan dalam folder history.")

elif current_tab == "History":
    st.title("History Simulasi")

    if not os.listdir(HISTORY_DIR):
        st.info("Belum ada data simulasi.")
    else:
        history_list = sorted(os.listdir(HISTORY_DIR), reverse=True)
        selected = st.selectbox("Pilih simulasi dari history:", history_list)

        sim_path = os.path.join(HISTORY_DIR, selected)
        config_path = os.path.join(sim_path, "config.txt")
        excel_path = os.path.join(sim_path, "output_simulasi.xlsx")

        with open(config_path, "r") as f:
            st.markdown("### Konfigurasi Simulasi:")
            st.code(f.read())

        if os.path.exists(excel_path):
            st.markdown("### Visualisasi Hasil Simulasi:")

            # Plot Time Series
            try:
                df_ts = pd.read_excel(excel_path, sheet_name="Time_Series_Analysis")
                fig1, ax1 = plt.subplots()
                df_ts.set_index('Timestamp')[['CBR_mean', 'SINR_mean']].plot(ax=ax1)
                ax1.set_title("CBR Mean & SINR Mean per Timestamp")
                st.pyplot(fig1)

                # Plot Detailed Results
                df_dr = pd.read_excel(excel_path, sheet_name="Detailed_Results")
                df_grouped = df_dr.groupby('Timestamp')[['Latency', 'PDR']].mean()
                fig2, ax2 = plt.subplots()
                df_grouped.plot(ax=ax2)
                ax2.set_title("Rata-rata Latency & PDR per Timestamp")
                st.pyplot(fig2)

            except Exception as e:
                st.error(f"Gagal membaca file Excel: {e}")
        else:
            st.warning("File hasil simulasi tidak ditemukan.")