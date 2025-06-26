import socket
import logging
import json
import struct
from datetime import datetime

LOG_RECEIVED_PATH = 'baseline/receive_data.log'
LOG_SENT_PATH = 'baseline/sent_data.log'

def log_data(log_path, data):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a') as log_file:
        log_file.write(f"[{timestamp}] {data}\n")

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
    print("received:", batch_data)
    
    responses = {}
    
    for veh_id, vehicle_data in batch_data.items():
        current_power = max(min(vehicle_data['transmissionPower'], 30.0), 1.0)
        current_beacon = max(min(vehicle_data['beaconRate'], 20.0), 1.0)
        snr = max(min(vehicle_data['SINR'], 50.0), 1.0)

        responses[veh_id] = {
            "transmissionPower": float(current_power),
            "beaconRate": float(current_beacon),
            "MCS": adjust_mcs_based_on_snr(snr),
        }

    response_data = json.dumps(responses).encode('utf-8')
    response_length = len(response_data)
    length_header = struct.pack('<I', response_length)
    conn.sendall(length_header)
    conn.sendall(response_data)
    formatted_response = json.dumps(responses)
    log_data(LOG_SENT_PATH, {formatted_response})
    print("sent:", formatted_response)

conn.close()
server.close()
logging.info("Server closed.")
