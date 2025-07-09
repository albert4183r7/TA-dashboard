import socket
import torch
import logging
import json
import os
import csv
import struct
import torch.serialization
from datetime import datetime
from SAC.sac_agent_vehicle import SACAgent

LOG_RECEIVED_PATH = 'SAC/logs/testing/receive_data.log'
LOG_DEBUG_ACTION_PATH = 'SAC/logs/testing/action.log'
LOG_SENT_PATH = 'SAC/logs/testing/sent_data.log'
PERFORMANCE_LOG_PATH = 'SAC/logs/testing/testing_metrics.csv'

def log_data(log_path, data):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a') as log_file:
        log_file.write(f"[{timestamp}] {data}\n")
        
def write_performance_metrics(timestamp, veh_id, cbr, snr, file_path=PERFORMANCE_LOG_PATH):
    file_exists = os.path.exists(file_path)
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'veh_id', 'CBR', 'SINR'])
        writer.writerow([timestamp, veh_id, cbr, snr])

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

logging.basicConfig(level=logging.INFO)

# Inisialisasi agen SAC
state_dim = 5  
action_dim = 2 
agent = SACAgent(state_dim, action_dim)

def load_model(agent, filename):
    """Load a trained SAC model from model folder."""
    try:
        # torch.serialization.add_safe_globals([SACAgent])  # Allowlist SACAgent for security
        checkpoint = torch.load(filename, map_location=agent.device, weights_only=False)
        
        # Handle dictionary-based checkpoint
        if isinstance(checkpoint, dict):
            logging.info("Loading dictionary-based checkpoint")
            required_keys = ['policy_state_dict', 'critic1_state_dict', 'critic2_state_dict']
            if not all(key in checkpoint for key in required_keys):
                raise KeyError(f"Checkpoint missing required keys: {required_keys}")
            
            agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            agent.q_net1.load_state_dict(checkpoint['critic1_state_dict'])
            agent.q_net2.load_state_dict(checkpoint['critic2_state_dict'])
            agent.target_q_net1.load_state_dict(checkpoint['critic1_target_state_dict'])
            agent.target_q_net2.load_state_dict(checkpoint['critic2_target_state_dict'])
            agent.log_alpha = checkpoint['log_alpha'].to(agent.device)
            agent.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            agent.q_optimizer1.load_state_dict(checkpoint['q_optimizer1_state_dict'])
            agent.q_optimizer2.load_state_dict(checkpoint['q_optimizer2_state_dict'])
            agent.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            agent.gamma = checkpoint['gamma']
            agent.tau = checkpoint['tau']
        
        # Handle full SACAgent object checkpoint
        elif isinstance(checkpoint, SACAgent):
            logging.info("Loading full SACAgent object checkpoint")
            agent.policy_net.load_state_dict(checkpoint.policy_net.state_dict())
            agent.q_net1.load_state_dict(checkpoint.q_net1.state_dict())
            agent.q_net2.load_state_dict(checkpoint.q_net2.state_dict())
            agent.target_q_net1.load_state_dict(checkpoint.target_q_net1.state_dict())
            agent.target_q_net2.load_state_dict(checkpoint.target_q_net2.state_dict())
            agent.log_alpha = checkpoint.log_alpha.to(agent.device)
            agent.policy_optimizer.load_state_dict(checkpoint.policy_optimizer.state_dict())
            agent.q_optimizer1.load_state_dict(checkpoint.q_optimizer1.state_dict())
            agent.q_optimizer2.load_state_dict(checkpoint.q_optimizer2.state_dict())
            agent.alpha_optimizer.load_state_dict(checkpoint.alpha_optimizer.state_dict())
            agent.gamma = checkpoint.gamma
            agent.tau = checkpoint.tau
        
        else:
            raise ValueError(f"Unsupported checkpoint type: {type(checkpoint)}")
        
        logging.info(f"Model successfully loaded from {filename}")
        return agent
    
    except Exception as e:
        logging.error(f"Failed to load model from {filename}: {str(e)}")
        raise

logging.basicConfig(level=logging.INFO)

model_path = "SAC/model/final_custom.pth"
if os.path.exists(model_path):
    agent = load_model(agent, model_path)

# Membuka soket untuk menerima data
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost", 5000))
server.listen(1)
logging.info("Listening on port 5000...")
conn, addr = server.accept()
logging.info(f"Connected by {addr}")


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
    
    responses = {}
    
    for veh_id, vehicle_data in batch_data.items():
        current_power = max(min(vehicle_data['transmissionPower'], 30.0), 1.0)
        current_beacon = max(min(vehicle_data['beaconRate'], 20.0), 1.0)
        cbr = max(min(vehicle_data['CBR'], 1.0), 0.0)
        snr = max(min(vehicle_data['SINR'], 50.0), 1.0)
        mcs = max(min(vehicle_data['MCS'], 10), 0)
        neighbors = vehicle_data['neighbors']
        timestamp = vehicle_data['timestamp']

        state = [current_power, current_beacon, cbr, neighbors, snr]
        
        write_performance_metrics(timestamp, veh_id, cbr, snr)
        
        action = agent.select_action(state)

        # normalized actions
        new_power = (action[0] + 1) / 2 * (30 - 10) + 10
        new_beacon = (action[1] + 1) / 2 * (20 - 10) + 10
        
        action_float = tuple(map(float, [new_power, new_beacon]))
        log_data(LOG_DEBUG_ACTION_PATH, {json.dumps(action_float, indent=4)})

        responses[veh_id] = {
            "transmissionPower": float(new_power),
            "beaconRate": float(new_beacon),
            "MCS": adjust_mcs_based_on_snr(snr),
        }

    response_data = json.dumps(responses).encode('utf-8')
    response_length = len(response_data)
    length_header = struct.pack('<I', response_length)
    conn.sendall(length_header)
    conn.sendall(response_data)
    formatted_response = json.dumps(responses)
    log_data(LOG_SENT_PATH, {formatted_response})

conn.close()
server.close()
logging.info("Server closed.")
