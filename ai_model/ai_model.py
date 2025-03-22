import torch
import torch.nn as nn

# Define AI Model
class GasLimitAI(nn.Module):
    def __init__(self):
        super(GasLimitAI, self).__init__()
        self.fc1 = nn.Linear(3, 32)  # 3 Inputs: Pending TXs, Gas Used, Current Gas Limit
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x)) * 30000000  # Scale output to gas range

# Load trained model
def load_model():
    model = GasLimitAI()
    model.load_state_dict(torch.load("ai-agent/ai_model.pth"))  # Load trained weights
    model.eval()
    return model

# Predict Gas Limit
def predict_gas_limit(pending_txs, gas_used, gas_limit):
    model = load_model()
    input_data = torch.tensor([[pending_txs, gas_used, gas_limit]], dtype=torch.float32)
    predicted_gas_limit = model(input_data).item()
    return int(predicted_gas_limit)

# Example Usage
if __name__ == "__main__":
    example_prediction = predict_gas_limit(50, 12000000, 15000000)
    print(f"Predicted Gas Limit: {example_prediction}")
