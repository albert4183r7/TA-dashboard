import socket
import torch
import logging
import json
import os
import csv
import struct
import numpy as np
from datetime import datetime
from SAC.sac_agent import SACAgent, save_replay_buffer, load_replay_buffer

LOG_RECEIVED_PATH = 'SAC/logs/training/receive_data.log'
LOG_SENT_PATH = 'SAC/logs/training/sent_data.log'
LOG_DEBUG_ACTION_PATH = 'SAC/logs/training/action.log'
PERFORMANCE_LOG_PATH = 'SAC/logs/training/performance_metrics.csv'
LOG_LOSS_PATH = 'SAC/logs/training/loss.csv'


def log_data(log_path, data):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a') as log_file:
        log_file.write(f"[{timestamp}] {data}\n")
        
def write_performance_metrics(timestamp, veh_id, cbr, neighbors, snr, reward, file_path=PERFORMANCE_LOG_PATH):
    file_exists = os.path.exists(file_path)
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'veh_id', 'CBR', 'neighbors', 'SNR', 'Reward'])
        writer.writerow([timestamp, veh_id, cbr, neighbors, snr, reward])

def write_record_loss(timestamp, actor_loss, critic1_loss, critic2_loss, alpha_loss, file_path=LOG_LOSS_PATH):
    file_exists = os.path.exists(file_path)
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'actor', 'Critic1', 'critic2', 'alpha'])
        writer.writerow([timestamp, actor_loss, critic1_loss, critic2_loss, alpha_loss])

def adjust_mcs_based_on_snr(snr):
    if 0 <= snr < 5:
        return 0
    elif 5 <= snr < 10:
        return 1
    elif 10 <= snr < 15:
        return 2
    elif 15 <= snr < 20:
        return 3
    elif 20 <= snr < 25:
        return 4
    elif 25 <= snr < 30:
        return 5
    elif 30 <= snr < 35:
        return 6
    elif 35 <= snr < 40:
        return 7
    elif 40 <= snr < 45:
        return 8
    elif 45 <= snr < 50:
        return 9
    elif snr >= 50:
        return 10
    else:
        return 0

def calculation_reward(cbr, sinr):
    """
    Menghitung reward berdasarkan kedekatan CBR dan SINR dengan target.
    
    Parameters:
    - cbr (float): Nilai Channel Busy Ratio saat ini (0 hingga 1).
    - sinr (float): Nilai SINR saat ini (dalam dB).
    - cbr_target (float): Nilai target CBR (default: 0.5).
    - sinr_target (float): Nilai target SINR (default: 20 dB).
    - cbr_weight (float): Bobot untuk komponen CBR (default: 0.5).
    - sinr_weight (float): Bobot untuk komponen SINR (default: 0.5).
    - max_reward (float): Nilai reward maksimum (default: 1.0).
    
    Returns:
    - reward (float): Nilai reward berdasarkan kedekatan dengan target.
    """
    # Normalisasi error CBR (dari 0 hingga 1)
    cbr_error = abs(cbr - 0.65) / max(0.65, 1 - 0.65)
    cbr_reward = 1.0 - cbr_error  # Semakin kecil error, semakin besar reward
    
    # Normalisasi error SINR (menggunakan fungsi eksponensial untuk skala dB)
    sinr_error = abs(sinr - 18)
    sinr_reward = np.exp(-sinr_error / 18)  # Eksponensial untuk sensitivitas pada SINR
    
    # Gabungkan reward dengan bobot
    total_reward = (0.5 * cbr_reward + 0.5 * sinr_reward) * 1.0
    
    # Pastikan reward berada dalam rentang [0, max_reward]
    return np.clip(total_reward, 0.0, 1.0)


logging.basicConfig(level=logging.INFO)

checkpoint_path = "SAC/model/latest_checkpoint.pth"
if os.path.exists(checkpoint_path):
    agent = torch.load(checkpoint_path)
    logging.info(f"Model successfully loaded from {checkpoint_path}")
else:
    agent = SACAgent(5, 2)
    logging.info("New model initialized.")
    
buffer_path = "SAC/model/replay_buffer.pkl"
if os.path.exists(buffer_path):
    load_replay_buffer(agent.replay_buffer, buffer_path)
    logging.info(f"Replay buffer successfully loaded from {buffer_path}")
else:
    logging.info("New Replay buffer initialized.")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost", 5000))
server.listen(1)
logging.info("Listening on port 5000...")
conn, addr = server.accept()
logging.info(f"Connected by {addr}")

prev_states = {}
prev_actions = {}

while True:
    length_header = conn.recv(4)
    if len(length_header) < 4:
        break
    msg_length = struct.unpack('<I', length_header)[0]
    data = conn.recv(msg_length)
    if len(data) < msg_length:
        break

    batch_data = json.loads(data.decode())
    log_data(LOG_RECEIVED_PATH, {json.dumps(batch_data)})
    
    for veh_id, vehicle_data in batch_data.items():
        current_power = max(min(vehicle_data['transmissionPower'], 30.0), 1.0)
        current_beacon = max(min(vehicle_data['beaconRate'], 20.0), 1.0)
        cbr = max(min(vehicle_data['CBR'], 1.0), 0.0)
        snr = max(min(vehicle_data['SINR'], 50.0), 1.0)
        mcs = max(min(vehicle_data['MCS'], 10), 0)
        neighbors = vehicle_data['neighbors']
        timestamp = vehicle_data['timestamp']

        current_state = [current_power, current_beacon, cbr, neighbors, snr]
        
        if timestamp > 0 and veh_id in prev_states:
            reward = calculation_reward(cbr, snr)
            write_performance_metrics(timestamp, veh_id, cbr, neighbors, snr, reward)
            agent.store_transition(prev_states[veh_id], prev_actions[veh_id], reward, current_state, False)
            
        prev_states[veh_id] = current_state
    
    # TRAINING THE PREVIOUS TIMESTAMP
    if timestamp > 0:
        for i in range(5):
            agent.train()
        save_replay_buffer(agent.replay_buffer, buffer_path)
        logging.info("Replay buffer saved to model/replay_buffer.pkl")
        torch.save(agent, 'SAC/model/latest_checkpoint.pth')
        logging.info("Model saved to model/latest_checkpoint.pth")
        policy_loss = agent.policy_loss
        q1_loss = agent.q1_loss
        q2_loss = agent.q2_loss
        alpha_loss =agent.alpha_loss
        write_record_loss(timestamp, policy_loss, q1_loss, q2_loss, alpha_loss)
        logging.info("Loss data saved to custom/logs/training/loss.csv")
    
    # =========================================================
    # AFTER TRAINING, START SELECT ACTION FOR CURRENT TIMESTAMP
    # =========================================================
    responses = {}
    
    for veh_id in prev_states:
        action = agent.select_action(prev_states[veh_id])
        
        # normalized actions
        new_power = (action[0] + 1) / 2 * (30 - 1) + 1
        new_beacon = (action[1] + 1) / 2 * (20 - 1) + 1
        
        prev_actions[veh_id] = [new_power, new_beacon]
        action_float = tuple(map(float, [new_power, new_beacon]))
        log_data(LOG_DEBUG_ACTION_PATH, {json.dumps(action_float, indent=4)})

        responses[veh_id] = {
            "transmissionPower": float(new_power),
            "beaconRate": float(new_beacon),
            "MCS": adjust_mcs_based_on_snr(prev_states[veh_id][3]),
        }

    response_data = json.dumps(responses).encode('utf-8')
    response_length = len(response_data)
    length_header = struct.pack('<I', response_length)
    conn.sendall(length_header)
    conn.sendall(response_data)
    formatted_response = json.dumps(responses)
    log_data(LOG_SENT_PATH, {formatted_response})

torch.save(agent, 'SAC/model/final_custom.pth')
logging.info("Model saved to SAC/model/final_custom.pth")
conn.close()
server.close()
logging.info("Server closed.")
