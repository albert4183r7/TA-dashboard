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

# Path logs dan model
LOG_RECEIVED_PATH = 'SAC/logs/training/receive_data.log'
LOG_SENT_PATH = 'SAC/logs/training/sent_data.log'
LOG_DEBUG_ACTION_PATH = 'SAC/logs/training/action.log'
PERFORMANCE_LOG_PATH = 'SAC/logs/training/performance_metrics.csv'
LOG_LOSS_PATH = 'SAC/logs/training/loss.csv'

BEST_MODEL_PATH = 'SAC/model/best_model.pth'
LATEST_MODEL_PATH = 'SAC/model/latest_checkpoint.pth'
BUFFER_PATH = 'SAC/model/replay_buffer.pkl'

# Setup logger
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
    # Target values and normalization
    cbr_target = 0.65
    sinr_target = 17.5
    
    # Gaussian rewards with adaptive widths
    cbr_reward = np.exp(-10 * (cbr - cbr_target)**2)
    sinr_reward = np.exp(-0.02 * (sinr - sinr_target)**2)
    
    # Balanced composite reward (normalized to [0,1])
    return float(0.5 * cbr_reward + 0.5 * sinr_reward)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load agent
if os.path.exists(LATEST_MODEL_PATH):
    agent = torch.load(LATEST_MODEL_PATH)
    logging.info(f"Model successfully loaded from {LATEST_MODEL_PATH}")
else:
    agent = SACAgent(5, 2)
    logging.info("New model initialized.")

# Load replay buffer
if os.path.exists(BUFFER_PATH):
    load_replay_buffer(agent.replay_buffer, BUFFER_PATH)
    logging.info(f"Replay buffer successfully loaded from {BUFFER_PATH}")
else:
    logging.info("New Replay buffer initialized.")

# Socket server setup
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost", 5000))
server.listen(1)
logging.info("Listening on port 5000...")
conn, addr = server.accept()
logging.info(f"Connected by {addr}")

# Best model tracking
best_avg_reward = float('-inf')
recent_rewards = []
REWARD_WINDOW_SIZE = 50  # Average of last N rewards

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
    log_data(LOG_RECEIVED_PATH, json.dumps(batch_data))

    for veh_id, vehicle_data in batch_data.items():
        current_power = max(min(vehicle_data['transmissionPower'], 30.0), 1.0)
        current_beacon = max(min(vehicle_data['beaconRate'], 25.0), 1.0)
        cbr = max(min(vehicle_data['CBR'], 1.0), 0.0)
        snr = max(min(vehicle_data['SINR'], 50.0), 1.0)
        mcs = max(min(vehicle_data['MCS'], 10), 0)
        neighbors = vehicle_data['neighbors']
        timestamp = vehicle_data['timestamp']

        current_state = [current_power, current_beacon, cbr, neighbors, snr]

        if timestamp > 0 and veh_id in prev_states:
            reward = calculation_reward(cbr, snr)
            recent_rewards.append(reward)
            if len(recent_rewards) > REWARD_WINDOW_SIZE:
                recent_rewards.pop(0)
            avg_reward = sum(recent_rewards) / len(recent_rewards)

            write_performance_metrics(timestamp, veh_id, cbr, neighbors, snr, reward)

            agent.store_transition(prev_states[veh_id], prev_actions[veh_id], reward, current_state, False)

        prev_states[veh_id] = current_state

    # TRAINING
    if timestamp > 0:
        agent.train()

        # Save latest model and buffer
        save_replay_buffer(agent.replay_buffer, BUFFER_PATH)
        logging.info(f"Replay buffer saved to {BUFFER_PATH}")
        torch.save(agent, LATEST_MODEL_PATH)
        logging.info(f"Latest model saved to {LATEST_MODEL_PATH}")

        # Record loss
        policy_loss = agent.policy_loss
        q1_loss = agent.q1_loss
        q2_loss = agent.q2_loss
        alpha_loss = agent.alpha_loss
        write_record_loss(timestamp, policy_loss, q1_loss, q2_loss, alpha_loss)
        logging.info(f"Loss data saved to {LOG_LOSS_PATH}")

        # Save best model jika avg_reward lebih tinggi
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save(agent, BEST_MODEL_PATH)
            logging.info(f"New best model saved to {BEST_MODEL_PATH} with avg reward: {best_avg_reward:.4f}")

    # SELECT ACTION
    responses = {}
    for veh_id in prev_states:
        action = agent.select_action(prev_states[veh_id])

        # Denormalisasi aksi
        new_power = (action[0] + 1) / 2 * (30 - 10) + 10
        new_beacon = (action[1] + 1) / 2 * (25 - 15) + 15

        prev_actions[veh_id] = [new_power, new_beacon]
        action_float = tuple(map(float, [new_power, new_beacon]))
        log_data(LOG_DEBUG_ACTION_PATH, json.dumps(action_float))

        responses[veh_id] = {
            "transmissionPower": float(new_power),
            "beaconRate": float(new_beacon),
            "MCS": adjust_mcs_based_on_snr(prev_states[veh_id][4]),
        }

    response_data = json.dumps(responses).encode('utf-8')
    response_length = len(response_data)
    length_header = struct.pack('<I', response_length)
    conn.sendall(length_header)
    conn.sendall(response_data)
    log_data(LOG_SENT_PATH, json.dumps(responses))

# Cleanup
torch.save(agent, 'SAC/model/final_custom.pth')
logging.info("Final model saved.")
conn.close()
server.close()
logging.info("Server closed.")