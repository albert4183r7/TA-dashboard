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
# ----------------------------------------------------


# =============================================================================
# Config & Hyper-parameters
# =============================================================================
HOST: str = "localhost"
PORT: int = 5000

# --- PERBAIKAN 4: Penyesuaian Hyper-parameter ---
LEARNING_RATE: float = 0.01      # Diturunkan untuk stabilitas
DISCOUNT_FACTOR: float = 0.99

# --- PERBAIKAN 1: Mengganti Epsilon Konstan dengan Epsilon Decay ---
# EPSILON: float = 0.1             # Baris ini tidak lagi digunakan
EPSILON_START: float = 1.0       # Mulai dengan 100% eksplorasi
EPSILON_DECAY: float = 0.9995    # Faktor peluruhan per batch
MIN_EPSILON: float = 0.01        # Batas bawah epsilon
# -----------------------------------------------------------------

CBR_TARGET: float = (0.6 + 0.7) / 2
SINR_TARGET = (15 + 20) / 2

# Discretisation bins 
POWER_BINS     = [5, 15, 25, 30]
BEACON_BINS    = [1, 5, 10, 20]
CBR_BINS       = [0.0, 0.3, 0.6, 1.0]
SINR_BINS      = [0, 5, 10, 15, 20, 25, 1e9]

# opsi untuk power dan beacon
ACTION_SPACE = [
    (3, 1),  # Action 0 : power +3 dBm, Beacon +1 Hz
    (3, -1), # Action 1 : power +3 dBm, Beacon -1 Hz
    (3, 0),  # Action 2 : power +3 dBm, Beacon no change
    (-3, 1), # Action 3 : power -3 dBm, Beacon +1 Hz
    (-3, -1),# Action 4 : power -3 dBm, Beacon -1 Hz
    (-3, 0), # Action 5 : power -3 dBm, Beacon no change
    (0, 1),  # Action 6 : power no change, Beacon +1 Hz
    (0, -1), # Action 7 : power no change, Beacon -1 Hz
    (0, 0),  # Action 8 : no change
]
ACTION_DIM = len(ACTION_SPACE)

# Accept alternative field names from client JSON
# (Timestamp ditambahkan untuk logging yang lebih baik)
FIELD_ALIASES = {
    "power": ["power", "transmissionPower", "current_power"],
    "beacon": ["beacon", "beaconRate", "current_beacon"],
    "cbr": ["cbr", "CBR", "channelBusyRatio"],
    "sinr": ["snr", "SNR", "signalToNoise", "SINR"],
    "timestamp": ["timestamp", "time"],
}

# =============================================================================
# Fungsi Helper 
# =============================================================================
def discretize(value: float, bins: list) -> int:
    """Return the index (0‑based) of *value* in *bins* for Q‑table lookup."""
    idx = np.digitize([value], bins, right=True)[0] #Cari indeks untuk value didalam bins
    return int(max(0, min(idx, len(bins) - 1))) # Memastikan indeks valid dalam range

def adjust_mcs_based_on_sinr(sinr: float) -> int:
    """Very coarse mapping from SNR (dB) → LTE‑V2X MCS index (0–7)."""
    if sinr > 25: return 7
    if sinr > 20: return 6
    if sinr > 15: return 5
    if sinr > 10: return 4
    if sinr > 5: return 3
    if sinr > 0: return 2
    return 1

# =============================================================================
# Q‑learning storage 
# =============================================================================
q_table: defaultdict[tuple, dict] = defaultdict(lambda: {i: 0.0 for i in range(ACTION_DIM)})

# =============================================================================
# Implementasi Server RL
# =============================================================================
class QLearningServerBatch:
    def __init__(self, host: str = HOST, port: int = PORT) -> None:
        self._setup_logging()
        self._ensure_csv_header()
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((host, port))
        self.server.listen(1)
        logging.info("Server listening on %s:%s", host, port)
        
        # Inisialisasi list untuk plotting dan epsilon dinamis ---
        self.sinr_list = []
        self.cbr_list = []
        self.reward_list = []
        self.q_delta_history = []
        self.epsilon = EPSILON_START
        self.should_shutdown = False
        # ----------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # Logging helpers
    # --------------------------------------------------------------------- 
    def _setup_logging(self):
        """Configure root logger to write both console and file."""
        fmt = "%(asctime)s [%(levelname)-5s] %(message)s"
        datefmt = "%H:%M:%S"

         # Console handler
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        
        # File handler (append)
        file = logging.FileHandler("train_fixed.log", mode="w", encoding="utf-8")
        file.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        logging.basicConfig(level=logging.INFO, handlers=[console, file])

    def _ensure_csv_header(self):
        """Create metrics.csv with header if it doesn't exist."""
        if not os.path.exists("metrics_fixed.csv"):
            with open("metrics_fixed.csv", "w", newline="") as f:
                csv.writer(f).writerow(["timestamp", "veh_id", "power", "beacon", "cbr", "sinr", "reward", "epsilon"])

    #fungsi _append_metrics untuk menerima timestamp
    def _append_metrics(self, veh_id: str, power: float, beacon: float, cbr: float, sinr: float, reward: float, timestamp: str, epsilon: float):
        with open("metrics_fixed.csv", "a", newline="") as f:
            csv.writer(f).writerow([timestamp, veh_id, power, beacon, cbr, sinr, reward, epsilon])

    # ---------------------------------------------------------------------
    # Inti Q-learning
    # ---------------------------------------------------------------------
    def _state_indices(self, state: list) -> tuple: #Fungsi discretize yang dipakai untuk konversi state menjadi index pada q-table lookup
        p, b, c, s = state
        return (
            discretize(p, POWER_BINS),
            discretize(b, BEACON_BINS),
            discretize(c, CBR_BINS),
            discretize(s, SINR_BINS),
        )

    @staticmethod
    def _reward(cbr: float, sinr: float) -> float:
        reward_cbr = -10 * (cbr - CBR_TARGET)**2
        reward_sinr = -0.1 * (sinr - SINR_TARGET)**2
        return reward_cbr + reward_sinr

   
    def _select_action(self, state_idx: tuple) -> int:
        if random.random() < self.epsilon:
            return random.choice(range(ACTION_DIM))
        
        # Eksploitasi: Pilih aksi dengan Q-value tertinggi untuk state saat ini
        q_values = q_table[state_idx]
        return max(q_values, key=q_values.get)
    # ----------------------------------------------------------------------

    def _update_q(self, old_idx: tuple, action: int, reward: float, new_idx: tuple):
        old_q = q_table[old_idx][action]
        max_next_q = max(q_table[new_idx].values())
        # --- PERBAIKAN: Rumus Q-update disederhanakan, fungsinya tetap sama ---
        new_q = old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q - old_q)
        q_table[old_idx][action] = new_q
    
    def save_q_table(self, filename="q_table_fixed.npy"):
        np.save(filename, dict(q_table), allow_pickle=True)
        logging.info("Saved Q-table to %s", filename)
    
    @staticmethod
    def _extract_field(d: dict, canonical: str, default=0):
        for alias in FIELD_ALIASES.get(canonical, []):
            if alias in d:
                return d[alias]
        if default is not None:
            return default
        raise KeyError(canonical)

    # =============================================================================

    # =============================================================================
    def _handle(self, conn: socket.socket, address: tuple):
        logging.info("Connected: %s:%s", *address)
        batch_counter = 0
        prev_states, prev_actions = {}, {}
                

        while True:
            try:
                
                # PERBAIKAN 2: Simpan Q-Table lama untuk menghitung konvergensi
                q_table_old = copy.deepcopy(q_table)

                # Terima data dari klien
                length_header = conn.recv(4)
                if not length_header: 
                    logging.info("Client closed connection gracefully. Server will shut down.")
                    self.should_shutdown = True # Atur flag untuk berhenti
                    break
                msg_length = struct.unpack('<I', length_header)[0]
                data = conn.recv(msg_length)
                if not data: break

                batch_data = json.loads(data.decode())
                batch_counter += 1
                logging.info(f"[BATCH {batch_counter}] Epsilon: {self.epsilon:.4f}")
                logging.info(f"Isi Batch Data:\n{json.dumps(batch_data, indent=2)}")

                responses = {}
                batch_rewards, batch_sinr, batch_cbr = [], [], []

                # --- Langkah 1: Update Q-Table berdasarkan pengalaman dari langkah SEBELUMNYA ---
                for vid, d in batch_data.items():
                    current_state = [
                        float(self._extract_field(d, "power")),
                        float(self._extract_field(d, "beacon")),
                        float(self._extract_field(d, "cbr")),
                        float(self._extract_field(d, "sinr")),
                    ]
                    current_idx = self._state_indices(current_state)
                    timestamp = self._extract_field(d, "timestamp", default="N/A")
                    
                    batch_cbr.append(current_state[2])
                    batch_sinr.append(current_state[3])

                    if vid in prev_states:
                        prev_idx = self._state_indices(prev_states[vid])
                        prev_action = prev_actions[vid]
                        reward = self._reward(current_state[2], current_state[3])
                        batch_rewards.append(reward)
                        self._update_q(prev_idx, prev_action, reward, current_idx)
                        self._append_metrics(vid, *current_state, reward, timestamp, self.epsilon)
                
                # --- Langkah 2: Pilih aksi BARU untuk langkah BERIKUTNYA ---
                for vid, d in batch_data.items():
                    current_state = [
                        float(self._extract_field(d, "power")),
                        float(self._extract_field(d, "beacon")),
                        float(self._extract_field(d, "cbr")),
                        float(self._extract_field(d, "sinr")),
                    ]
                    current_idx = self._state_indices(current_state)
                    action = self._select_action(current_idx)
                    
                    prev_states[vid], prev_actions[vid] = current_state, action

                    delta_power, delta_beacon = ACTION_SPACE[action]
                    new_power = max(POWER_BINS[0], min(current_state[0] + delta_power, POWER_BINS[-1]))
                    new_beacon = max(BEACON_BINS[0], min(current_state[1] + delta_beacon, BEACON_BINS[-1]))
                    
                    responses[vid] = {
                        "transmissionPower": float(new_power),
                        "beaconRate": float(new_beacon),
                        "MCS": adjust_mcs_based_on_sinr(current_state[3])
                    }

                # --- Langkah 3: Hitung Metrik & Lakukan Epsilon Decay ---
                total_difference, entry_count = 0.0, 0
                for state_idx, action_dict in q_table.items():
                    if state_idx in q_table_old:
                        for action_idx, q_value in action_dict.items():
                            old_q_value = q_table_old[state_idx].get(action_idx, 0.0)
                            total_difference += abs(q_value - old_q_value)
                            entry_count += 1
                mean_delta = total_difference / entry_count if entry_count > 0 else 0.0
                self.q_delta_history.append(mean_delta)

                self.reward_list.append(sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0)
                self.sinr_list.append(sum(batch_sinr) / len(batch_sinr) if batch_sinr else 0)
                self.cbr_list.append(sum(batch_cbr) / len(batch_cbr) if batch_cbr else 0)
                
                self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

                # --- Langkah 4: Simpan & Plot secara berkala ---
                if batch_counter > 0 and batch_counter % 100 == 0:
                    plot_metrics(self.sinr_list, self.cbr_list, self.reward_list, self.q_delta_history, batch_counter)
                    logging.info(f"Metrics plot saved at batch {batch_counter}")
                    self.save_q_table(f"q_table_checkpoint_fixed_{batch_counter}.npy")

                # Kirim balasan ke klien
                response_data = json.dumps(responses).encode('utf-8')
                conn.sendall(struct.pack('<I', len(response_data)))
                conn.sendall(response_data)
                
            except (ConnectionResetError, BrokenPipeError):
                logging.warning("Client disconnected.")
                self.should_shutdown = True # Atur flag untuk berhenti
                break
            except Exception as e:
                logging.error(f"An error occurred: {e}", exc_info=True)
                break
        
        logging.info("Closing connection for %s:%s", *address)
        conn.close()

    def serve_forever(self):
        try:
            # Loop ini sekarang akan berhenti setelah satu klien selesai
            # karena _handle akan mengatur self.should_shutdown menjadi True.
            while not self.should_shutdown:
                logging.info("Waiting for simulation client to connect...")
                conn, addr = self.server.accept()
                self._handle(conn, addr) # Metode ini akan berjalan sampai klien disconnect

            logging.info("Shutdown signal received from client disconnection.")

        except KeyboardInterrupt:
            logging.info("Server stopped by user (Ctrl+C).")
        finally:
            # berhenti normal maupun karena Ctrl+C
            logging.info("Saving final model...")
            self.save_q_table()
            self.server.close() # Menutup socket server utama
            logging.info("Server has been shut down completely.")

# =============================================================================
# PERBAIKAN 3: FUNGSI PLOTTING 
# =============================================================================
def plot_metrics(sinr_list, cbr_list, reward_list, q_delta_list, batch_counter):
    batches = range(1, len(sinr_list) + 1)
    plt.figure(figsize=(12, 10))

    plt.subplot(4, 1, 1)
    plt.plot(batches, sinr_list, label="Average SINR", color='blue')
    plt.ylabel("Average SINR (dB)")
    plt.title(f"Performance Metrics up to Batch {batch_counter}")
    plt.legend(); plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(batches, cbr_list, label="Average CBR", color='orange')
    plt.ylabel("Average CBR"); plt.axhline(y=CBR_TARGET, color='r', linestyle='--', label=f'Target {CBR_TARGET:.2f}')
    plt.legend(); plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(batches, reward_list, label="Average Reward", color='green')
    plt.ylabel("Reward"); plt.ylim(bottom=min(reward_list)-5 if reward_list else -40, top=1)
    plt.legend(); plt.grid(True)
    
    plt.subplot(4, 1, 4)
    plt.plot(batches, q_delta_list, label="Q-Table Delta", color='red')
    plt.ylabel("Mean Q-Value Change")
    plt.xlabel("Batch Number")
    plt.legend(); plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(f"metrics_plot_fixed_{batch_counter}.png")
    plt.close()

# =============================================================================
# Titik Masuk Program 
# =============================================================================
if __name__ == "__main__":
    QLearningServerBatch().serve_forever()
