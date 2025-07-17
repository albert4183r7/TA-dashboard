# 🚗 Joint Optimization of Beacon Rate and Transmission Power For Vehicle-To-Vehicle (V2V) Communication Using Reinforcement Learning (RL) Algorithm in Highway Line-Of-Sight (LOS) Scenario

## Overview
This project aims to optimize beacon rate and transmission power in Vehicle-to-Vehicle (V2V) IEEE802.11bd standard communication in  highway LOS scenario using Reinforcement Learning (RL) techniques: Q-learning and Soft Actor-Critic (SAC). The goal is to maximized channel uitilization and beaconing quality by maintaining the Channel Busy Ratio (CBR) and SINR (Signal to Interference plus Noise Ratio) within target ranges:
- CBR : 0.6 - 0.7
- SINR : 15 dB - 20 dB

## 📁 Project Structure
```
.
├── baseline/                  # Baseline (non-RL) simulation scripts
│   ├── logs/
│   └── run.py
├── Q-learning/                # Q-learning implementation
│   ├── logs/
│   ├── model/
│   ├── train.py
│   └── test.py
├── SAC/                       # Soft Actor-Critic implementation
│   ├── logs/
│   ├── model/
│   ├── sac_agent.py
│   ├── train.py
│   └── test.py
├── simulasi_vanet.py          # From external repository (see below)
├── app.py                     # Streamlit dashboard application
├── requirements.txt           # Python dependencies
└── README.md              
```

## 📥 External Dependency: Mobility Simulation Script
The file simulasi_vanet.py is sourced from an external GitHub repository:
👉 https://github.com/galihnnk/CANVAS-VANET

### Clone That Repository:
```bash 
git clone https://github.com/galihnnk/CANVAS-VANET.git
```

### Move Required Files:
After cloning, copy CANVAS.py as simulasi_vanet.py from that repository into your current project folder:
```bash 
cp CANVAS-VANET/CANVAS.py ./simulasi_vanet.py
```
Or adjust the subprocess call in ```app.py``` so it points directly to the cloned folder (without copying).


## ▶️ How to Run
1. Activate your virtual environment (if applicable):
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch the dashboard using Streamlit:
```bash
streamlit run app.py
```

4. Open browser and go to: http://localhost:8501

## 📤 Upload FCD (Floating Car Data)
- Data must follow SUMO’s Floating Car Data (FCD) format.
- User can select:
  - EnableRL: True or False
  - Mode: Training / Testing
  - Algorithm: Q-learning / SAC
- After uploading the file and selecting options, click "Run Simulation".
- Output will be saved in the simulasi_history/ folder containing:
   - config.txt: Simulation configuration
   - output_simulasi.xlsx: Metrics like CBR, SINR, PDR, latency, beacon rate, tx_power, etc.

## 📊 Dashboard & History
- Home Tab: Upload file and start simulation.
- The History section displays previous simulations configuration and charts (CBR, SINR, Latency, PDR).

## 🔌 Socket Server Connection
- This project communicates with an external socket server running at localhost:5000.
- Make sure the socket server is running before starting simulations.
- If running on a different host or port, update the scripts inside:
- ```simulasi_vanet.py```
- ```Q-learning/train.py``` and ```test.py```
- ```SAC/train.py``` and ```test.py```

## ✅ Notes
- Ensure the socket server is running at ```localhost:5000``` or adjust the IP as needed.
- Use valid SUMO-formatted FCD data.
- Simulation outputs can be exported and further analyzed.

## 🛠️ Troubleshooting
- Missing Output File:
Check if simulasi_vanet.py ran successfully. Logs are saved in respective simulasi_history/ folders.

- Socket Server Connection Refused:
Make sure no firewall or port conflict exists. Confirm the server is running.

- Incorrect Path Error:
Adjust cwd parameter when calling subprocess.run() in app.py if file locations differ.

## 👨‍💻 Author & Contact
- Developed as part of academic research on vehicular communication systems.
- For inquiries, suggestions, or collaboration:
   - Name: Albert
   - Email: albertlie8338@gmail.com
   - GitHub: github.com/albert4183r7
   - Institution: Bandung Institute of Technologu