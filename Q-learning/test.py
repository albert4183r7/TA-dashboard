import socket
import struct
import json
import logging
import csv
import os
import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import copy
import argparse # <-- BARU: Untuk menerima path model dari command line

# =============================================================================
# Konfigurasi Environment (Sama seperti training)
# =============================================================================
HOST: str = "localhost"
PORT: int = 5000

CBR_TARGET: float = (0.6 + 0.7) / 2
SINR_TARGET = (15 + 20) / 2

# Discretisation bins
POWER_BINS     = [5, 15, 25, 30]
BEACON_BINS    = [1, 5, 10, 20]
CBR_BINS       = [0.0, 0.3, 0.6, 1.0]
SINR_BINS      = [0, 5, 10, 15, 20, 25, 1e9]

# Opsi untuk power dan beacon
ACTION_SPACE = [
    (3, 1), (3, -1), (3, 0),
    (-3, 1), (-3, -1), (-3, 0),
    (0, 1), (0, -1), (0, 0),
]
ACTION_DIM = len(ACTION_SPACE)

FIELD_ALIASES = {
    "power": ["power", "transmissionPower", "current_power"],
    "beacon": ["beacon", "beaconRate", "current_beacon"],
    "cbr": ["cbr", "CBR", "channelBusyRatio"],
    "sinr": ["snr", "SNR", "signalToNoise", "SINR"],
    "timestamp": ["timestamp", "time"],
}

# --- DIHAPUS ---
# Hyper-parameter training seperti LEARNING_RATE, DISCOUNT_FACTOR, EPSILON_* tidak diperlukan.

# =============================================================================
# Q‑learning storage
# =============================================================================
q_table: defaultdict[tuple, dict] = defaultdict(lambda: {i: 0.0 for i in range(ACTION_DIM)})

# =============================================================================
# Fungsi Helper (Tidak ada perubahan)
# =============================================================================
def discretize(value: float, bins: list) -> int:
    idx = np.digitize([value], bins, right=True)[0]
    return int(max(0, min(idx, len(bins) - 1)))

def adjust_mcs_based_on_sinr(sinr: float) -> int:
    if sinr > 25: return 7
    if sinr > 20: return 6
    if sinr > 15: return 5
    if sinr > 10: return 4
    if sinr > 5: return 3
    if sinr > 0: return 2
    return 1

# =============================================================================
# Implementasi Server TESTING
# =============================================================================
class QLearningTestServer:
    def __init__(self, model_path: str) -> None:
        self.should_shutdown = False
        self.epsilon = 0.0  # <-- PENTING: Epsilon = 0 untuk 100% eksploitasi (tidak ada aksi acak)

        self._setup_logging()
        self._ensure_csv_header()
        
        self.load_q_table(model_path) # <-- BARU: Langsung muat model saat inisialisasi
        if self.should_shutdown: # Jika model gagal dimuat, jangan lanjutkan
            return

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((HOST, PORT))
        self.server.listen(1)
        logging.info("TESTING Server listening on %s:%s", HOST, PORT)
        
        self.sinr_list = []
        self.cbr_list = []
        self.reward_list = []

    def _setup_logging(self):
        fmt = "%(asctime)s [%(levelname)-5s] [TESTING] %(message)s"
        datefmt = "%H:%M:%S"
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        file_handler = logging.FileHandler("test.log", mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        logging.basicConfig(level=logging.INFO, handlers=[console, file_handler])

    def _ensure_csv_header(self):
        if os.path.exists("metrics_test.csv"):
            os.remove("metrics_test.csv") # Hapus file lama agar hasil tes selalu baru
        with open("metrics_test.csv", "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "veh_id", "power", "beacon", "cbr", "sinr", "reward"])

    def _append_metrics(self, veh_id: str, power: float, beacon: float, cbr: float, sinr: float, reward: float, timestamp: str):
        with open("metrics_test.csv", "a", newline="") as f:
            csv.writer(f).writerow([timestamp, veh_id, power, beacon, cbr, sinr, reward])

    def load_q_table(self, filename: str): # <-- BARU: Fungsi untuk memuat model
        global q_table
        try:
            loaded_dict = np.load(filename, allow_pickle=True).item()
            temp_q_table = defaultdict(lambda: {i: 0.0 for i in range(ACTION_DIM)})
            temp_q_table.update(loaded_dict)
            q_table = temp_q_table
            logging.info("Successfully loaded Q-table from %s", filename)
        except FileNotFoundError:
            logging.error("MODEL NOT FOUND at %s. Cannot run test.", filename)
            self.should_shutdown = True

    def _state_indices(self, state: list) -> tuple:
        p, b, c, s = state
        return (discretize(p, POWER_BINS), discretize(b, BEACON_BINS), discretize(c, CBR_BINS), discretize(s, SINR_BINS))

    @staticmethod
    def _reward(cbr: float, sinr: float) -> float:
        reward_cbr = -10 * (cbr - CBR_TARGET)**2
        reward_sinr = -0.1 * (sinr - SINR_TARGET)**2
        return reward_cbr + reward_sinr

    def _select_action(self, state_idx: tuple) -> int:
        # Karena epsilon = 0, ini akan selalu memilih aksi terbaik (eksploitasi)
        if random.random() < self.epsilon:
            return random.choice(range(ACTION_DIM))
        q_values = q_table[state_idx]
        if not q_values: # Jika state belum pernah ditemui
            return random.choice(range(ACTION_DIM)) # Pilih aksi acak
        return max(q_values, key=q_values.get)
    
    # --- DIHAPUS ---
    # Fungsi _update_q() dan save_q_table() tidak diperlukan dalam mode testing.

    @staticmethod
    def _extract_field(d: dict, canonical: str, default=0):
        for alias in FIELD_ALIASES.get(canonical, []):
            if alias in d:
                return d[alias]
        if default is not None:
            return default
        raise KeyError(canonical)

    def _handle(self, conn: socket.socket, address: tuple):
        logging.info("Connected for testing: %s:%s", *address)
        batch_counter = 0
        prev_states = {} # Hanya butuh state sebelumnya untuk menghitung reward

        while True:
            try:
                length_header = conn.recv(4)
                if not length_header: 
                    logging.info("Client closed connection. Test finished.")
                    break
                msg_length = struct.unpack('<I', length_header)[0]
                data = conn.recv(msg_length)
                if not data: break

                batch_data = json.loads(data.decode())
                batch_counter += 1
                logging.info(f"[BATCH {batch_counter}] Running test...")

                responses = {}
                batch_rewards, batch_sinr, batch_cbr = [], [], []

                for vid, d in batch_data.items():
                    current_state = [
                        float(self._extract_field(d, "power")),
                        float(self._extract_field(d, "beacon")),
                        float(self._extract_field(d, "cbr")),
                        float(self._extract_field(d, "sinr")),
                    ]
                    timestamp = self._extract_field(d, "timestamp", default="N/A")
                    batch_cbr.append(current_state[2])
                    batch_sinr.append(current_state[3])

                    if vid in prev_states:
                        reward = self._reward(current_state[2], current_state[3])
                        batch_rewards.append(reward)
                        self._append_metrics(vid, *current_state, reward, timestamp)
                
                for vid, d in batch_data.items():
                    current_state = [
                        float(self._extract_field(d, "power")),
                        float(self._extract_field(d, "beacon")),
                        float(self._extract_field(d, "cbr")),
                        float(self._extract_field(d, "sinr")),
                    ]
                    current_idx = self._state_indices(current_state)
                    action = self._select_action(current_idx)
                    
                    prev_states[vid] = current_state # Simpan state untuk iterasi berikutnya

                    delta_power, delta_beacon = ACTION_SPACE[action]
                    # Tentukan batasan clip
                    POWER_MIN, POWER_MAX = 20, 33
                    BEACON_MIN, BEACON_MAX = 10, 20

                    # Hitung nilai baru berdasarkan aksi inkremental
                    new_power_candidate = current_state[0] + delta_power
                    new_beacon_candidate = current_state[1] + delta_beacon

                    # Gunakan np.clip untuk memaksa nilai berada dalam rentang
                    new_power = np.clip(new_power_candidate, POWER_MIN, POWER_MAX)
                    new_beacon = np.clip(new_beacon_candidate, BEACON_MIN, BEACON_MAX)
                    
                    responses[vid] = {
                        "transmissionPower": float(new_power),
                        "beaconRate": float(new_beacon),
                        "MCS": adjust_mcs_based_on_sinr(current_state[3])
                    }

                self.reward_list.append(sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0)
                self.sinr_list.append(sum(batch_sinr) / len(batch_sinr) if batch_sinr else 0)
                self.cbr_list.append(sum(batch_cbr) / len(batch_cbr) if batch_cbr else 0)
                
                if batch_counter > 0 and batch_counter % 100 == 0:
                    plot_test_metrics(self.sinr_list, self.cbr_list, self.reward_list, batch_counter)
                    logging.info(f"Test metrics plot saved at batch {batch_counter}")

                response_data = json.dumps(responses).encode('utf-8')
                conn.sendall(struct.pack('<I', len(response_data)))
                conn.sendall(response_data)
                
            except (ConnectionResetError, BrokenPipeError):
                logging.warning("Client disconnected.")
                break
            except Exception as e:
                logging.error(f"An error occurred: {e}", exc_info=True)
                break
        
        logging.info("Closing test connection for %s:%s", *address)
        conn.close()
        self.should_shutdown = True

    def serve_forever(self):
        if self.should_shutdown: # Jika model tidak ditemukan
             logging.error("Server cannot start because model was not loaded.")
             return
        try:
            while not self.should_shutdown:
                logging.info("Waiting for simulation client to connect for testing...")
                conn, addr = self.server.accept()
                self._handle(conn, addr)
        except KeyboardInterrupt:
            logging.info("Test server stopped by user (Ctrl+C).")
        finally:
            if self.sinr_list:
                plot_test_metrics(self.sinr_list, self.cbr_list, self.reward_list, 'final')
                logging.info("Final test metrics plot saved.")
            self.server.close()
            logging.info("Test server has been shut down.")

# =============================================================================
# Fungsi Plotting untuk Testing
# =============================================================================
def plot_test_metrics(sinr_list, cbr_list, reward_list, batch_counter):
    batches = range(1, len(sinr_list) + 1)
    plt.figure(figsize=(12, 8)) # Ukuran disesuaikan karena hanya 3 plot
    plt.suptitle(f"TESTING Performance Metrics up to Batch {batch_counter}", fontsize=16)

    plt.subplot(3, 1, 1)
    plt.plot(batches, sinr_list, label="Average SINR", color='blue')
    plt.ylabel("Average SINR (dB)"); plt.legend(); plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(batches, cbr_list, label="Average CBR", color='orange')
    plt.ylabel("Average CBR"); plt.axhline(y=CBR_TARGET, color='r', linestyle='--', label=f'Target {CBR_TARGET:.2f}')
    plt.legend(); plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(batches, reward_list, label="Average Reward", color='green')
    plt.ylabel("Reward"); plt.ylim(bottom=min(reward_list)-5 if reward_list else -40, top=1)
    plt.xlabel("Batch Number"); plt.legend(); plt.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"metrics_plot_test_{batch_counter}.png")
    plt.close()

# =============================================================================
# Titik Masuk Program
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Q-Learning TEST Server.")
    parser.add_argument(
        'model_path', 
        type=str, 
        help="Path to the pre-trained Q-table .npy file to be tested."
    )
    args = parser.parse_args()

    server = QLearningTestServer(model_path=args.model_path)
    server.serve_forever()
