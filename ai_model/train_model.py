import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from web3 import Web3

# Define AI Model
class GasLimitAI(nn.Module):
    def __init__(self):
        super(GasLimitAI, self).__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x)) * 30000000  # Scale output to gas range

# Reward Function (Encourage Lower Pending TXs)
def reward_function(pending_txs, failed_txs):
    return 10 if failed_txs == 0 else -1 * pending_txs

# Connect to Ethereum via Infura
INFURA_URL = "https://mainnet.infura.io/v3/b5f2559745fc404a8b630669565301b1"
web3 = Web3(Web3.HTTPProvider(INFURA_URL))

# Get Blockchain Data
def get_blockchain_data():
    latest_block = web3.eth.get_block('latest')
    pending_txs = web3.eth.get_block_transaction_count(latest_block.number)
    gas_used = latest_block.gasUsed
    gas_limit = latest_block.gasLimit
    return pending_txs, gas_used, gas_limit

# Train the AI Model
def train_ai(epochs=1000, lr=0.001):  # Reduced epochs for faster retraining
    model = GasLimitAI()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        pending_txs, gas_used, gas_limit = get_blockchain_data()
        failed_txs = np.random.randint(0, 10)  # Simulated failed transactions

        input_data = torch.tensor([[pending_txs, gas_used, gas_limit]], dtype=torch.float32)
        target = torch.tensor([[gas_limit]], dtype=torch.float32)  # Ideal gas limit (temporary)

        # Forward pass
        predicted = model(input_data)
        loss = criterion(predicted, target)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        reward = reward_function(pending_txs, failed_txs)

        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Reward={reward}")

    # Ensure the directory exists before saving the model
    save_path = "ai-agent/ai_model.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully at {save_path}")


# Run training
if __name__ == "__main__":
    train_ai()
