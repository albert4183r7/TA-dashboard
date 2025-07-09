# 🚗 Reinforcement Learning-Based Optimization of Beacon Rate and Transmission Power for V2V Communication
This project aims to optimize beacon rate , transmission power (tx_power) , and MCS (Modulation and Coding Scheme) in Vehicle-to-Vehicle (V2V) communication using Reinforcement Learning (RL) techniques: Q-learning and Soft Actor-Critic (SAC) . The goal is to maintain the Channel Busy Ratio (CBR) and SINR (Signal to Interference plus Noise Ratio) within target ranges:
CBR : 0.6 - 0.7
SINR : 15 dB - 20 dB

## 📁 Project Structure
.
├── baseline/
│   └── logs/                  # Baseline simulation logs
├── Q-learning/
│   ├── logs/
│   ├── model/                 # Q-learning trained models
│   ├── train.py               # Q-learning training script
│   └── test.py                # Q-learning testing script
├── SAC/
│   ├── logs/
│   │   ├── training/          # SAC training logs
│   │   └── testing/           # SAC testing logs
│   ├── model/                 # SAC trained models
│   ├── sac_agent.py           # SAC agent definition
│   ├── train.py               # SAC training script
│   └── test.py                # SAC testing script
├── simulasi_vanet.py          # Mobility simulation script
├── app.py                     # Streamlit dashboard application
├── plot.py                    # Plotting script for results
├── requirements.txt           # Python dependencies
└── README.md                  # This documentation

## 🧰 Requirements
Install required packages:
```bash
pip install -r requirements.txt
```

## ▶️ How to Run
1. Activate your virtual environment (if applicable):
```bash
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

2. Launch the dashboard using Streamlit:
```bash
streamlit run app.py
```

3. Open browser and go to: http://localhost:8501

## 📤 Upload FCD (Floating Car Data)
- Data must follow SUMO’s Floating Car Data (FCD) format.
- User can select simulation mode: Baseline , Q-learning , or SAC .
🏃‍♂️ Run Simulation
- After uploading the file and selecting a mode, click "Run Simulation".
- Output will be saved under the hasil_simulasi/ folder containing:
   - config.txt: Simulation configuration
   - output_simulasi.xlsx: Metrics like CBR, SINR, beacon rate, tx_power, etc.

## 📊 Dashboard & History
- The History section displays previous simulations.
- Interactive charts are available to analyze simulation performance.

## 🔌 Socket Server Connection
- Make sure the socket server is running at localhost:5000.
- If using a different IP/host, update the relevant scripts accordingly.

## 🎯 Optimization Techniques
1. Q-learning
- Tabular reinforcement learning approach.
- Suitable for small state-action spaces.
- Optimizes beacon rate and tx_power.
2. Soft Actor-Critic (SAC)
- Off-policy deep reinforcement learning algorithm.
- Enables generalization across mobility scenarios.
- Handles batch input from 44 vehicles simultaneously.

## 📈 Result Visualization
Dynamic plots displayed on dashboard:
- CBR and SINR over time
- Trends of beacon rate, transmission power, and MCS


## ✅ Notes
Ensure the socket server is running at localhost:5000 or adjust the IP as needed.
Use valid SUMO-formatted FCD data for optimal results.
Simulation outputs can be exported and further analyzed.

## 🧑‍💻 License & Contact
MIT License
For questions or collaboration, contact [email@example.com ]