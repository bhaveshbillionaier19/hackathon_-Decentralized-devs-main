# Autonomous Traffic Manager for Blockchain Networks

## 🚀 Overview
The **Autonomous Traffic Manager for Blockchain Networks** is a decentralized system that optimizes blockchain network performance using AI-driven gas price estimation and eBPF-based traffic filtering. The project dynamically adjusts block gas limits, prevents spam/DoS attacks, and efficiently distributes network traffic.

## 🏗 Tech Stack
- **AI**: PyTorch, TensorFlow (Reinforcement Learning models)
- **Blockchain**: Solidity, Hardhat, Ethereum
- **eBPF**: bcc, libbpf, Cilium (for efficient traffic filtering & load balancing)
- **Frontend**: React.js
- **Backend**: Flask(API)

---
### Key Features
- **AI Agent**: Utilizes Reinforcement Learning (PyTorch/TensorFlow) to optimize gas price dynamically.
- **Smart Contracts**: Implements `NetworkManager.sol` to manage network settings.
- **eBPF Load Balancing**: Uses `bcc`, `libbpf`, and `Cilium` for efficient node distribution.
- **Frontend (React.js)**: Web3-enabled UI for managing network parameters and viewing gas price estimates.

## 📂 Project Structure
```
📦 Autonomous-Traffic-Manager
├── 📂 contract/             # Contains NetworkManager.sol
├── 📂 src/                  # React frontend (App.js, components, etc.)
├── 📂 ai_agent/             # AI model training & inference code
├── 📂 ebpf/                 # eBPF scripts for network filtering
├── 📂 scripts/              # Deployment & automation scripts
├── 📜 hardhat.config.js      # Hardhat blockchain setup
├── 📜 README.md             # Project documentation
└── 📜 package.json          # Dependencies
```

---

## 🧠 AI Agent
The AI agent monitors and optimizes blockchain traffic using Reinforcement Learning:
1. **Observes network metrics** (gas fees, transaction volume, congestion levels).
2. **Predicts optimal parameters** (block gas limits, node distribution).
3. **Adjusts network configurations** dynamically.
4. **Uses reward functions** to optimize efficiency and security.

### 🏗 AI Model Training
- **Input**: Blockchain network data (transactions, gas fees, mempool state, node load)
- **Algorithm**: Reinforcement Learning (DQN/PPO)
- **Training Frameworks**: PyTorch / TensorFlow

### ⚡ AI Integration with Blockchain
- Smart contract interacts with AI API for real-time adjustments.
- AI agent sends optimized gas limits & node distribution strategies.

---

## ⚙️ eBPF Load Balancing & Security
- **Traffic Filtering**: eBPF filters spam/DoS traffic at the kernel level.
- **Node Load Balancing**: Dynamically redistributes transaction load across nodes.
- **Tooling**: Uses `bcc`, `libbpf`, and `Cilium`.

---

## Usage
### 1. Connect Wallet
- Open the web app.
- Click **Connect Wallet** to link MetaMask.

### 2. View Gas Price Prediction
- The AI agent continuously updates the optimal gas price.
- View the estimated value on the dashboard.

### 3. Adjust Network Parameters
- Modify gas limits and transaction fees.
- Submit changes via the smart contract.

### 4. Monitor Traffic Filtering (eBPF)
- Run the eBPF module:
```bash
cd eBPF
sudo ./start_filtering.sh
```
- Monitor logs for blocked transactions.

## 🚀 Deployment
### 1️⃣ Smart Contract Deployment
```sh
cd contract/
npm install
npx hardhat deploy
```

### 2️⃣ AI Agent Setup
```sh
cd ai_agent/
pip install -r requirements.txt
python train_model.py  # Train the RL model
python ai_api.py  # Run inference
```

### 3️⃣ eBPF Activation
```sh
cd ebpf/
sudo ./setup_ebpf.sh
```

### 4️⃣ Frontend
```sh
npm install
npm start
```

---

## 🛠 Future Improvements
- Enhance AI model efficiency with better reward shaping.
- Expand eBPF filtering rules for advanced security.
- Implement multi-chain compatibility.

---

## 📜 License
MIT License © 2025 Autonomous Traffic Manager Team

