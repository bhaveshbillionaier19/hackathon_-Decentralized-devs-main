import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ai_model import BlockchainOptimizerAI, NetworkState, determine_network_state

# ✅ Reward Function for Gas Limit Optimization
def reward_gas_limit(gas_limit, pending_tx_count, block_utilization):
    """
    Reward function for gas limit optimization:
    - Rewards higher gas limits during low congestion
    - Rewards lower gas limits during high congestion
    - Rewards gas limits in the sweet spot (28M-35M)
    """
    # Determine network state
    network_state = determine_network_state(pending_tx_count, block_utilization)
    
    # Initialize reward
    reward = 0
    
    # Check if gas limit is in reasonable range
    if 28000000 <= gas_limit <= 35000000:
        reward += 5  # Base reward for staying in reasonable range
    else:
        reward -= 5  # Penalty for unreasonable gas limits
    
    # Network state specific rewards
    if network_state == NetworkState.LOW:
        # In low congestion, reward higher gas limits for better throughput
        normalized_gas = (gas_limit - 28000000) / (35000000 - 28000000)  # 0-1 scale
        reward += normalized_gas * 5  # Max +5 reward for higher gas limit
    elif network_state == NetworkState.HIGH:
        # In high congestion, reward lower gas limits for stability
        normalized_gas = 1 - (gas_limit - 28000000) / (35000000 - 28000000)  # 1-0 scale
        reward += normalized_gas * 5  # Max +5 reward for lower gas limit
    
    return reward

# ✅ Reward Function for Fee Estimation
def reward_fee_estimation(fees, pending_tx_count, block_utilization):
    """
    Reward function for fee estimation:
    - Rewards reasonable fee spreads between slow/medium/fast
    - Rewards lower fees during low congestion
    - Rewards higher fees during high congestion
    """
    # Determine network state
    network_state = determine_network_state(pending_tx_count, block_utilization)
    
    # Initialize reward
    reward = 0
    
    # Extract fees
    slow_fee, medium_fee, fast_fee = fees
    
    # Check for correct ordering (slow < medium < fast)
    if slow_fee <= medium_fee <= fast_fee:
        reward += 3
    else:
        reward -= 10  # Strong penalty for incorrect ordering
    
    # Check fee spread
    if fast_fee > slow_fee * 1.5:  # Fast should be significantly higher than slow
        reward += 2
    
    # Network state specific rewards
    if network_state == NetworkState.LOW:
        # In low congestion, reward lower fees
        avg_fee = (slow_fee + medium_fee + fast_fee) / 3
        if avg_fee < 20:  # If average fee is low
            reward += 3
    elif network_state == NetworkState.HIGH:
        # In high congestion, reward higher fees for priority
        if fast_fee > 30:  # If fast fee is high enough for priority
            reward += 3
    
    return reward

# ✅ Reward Function for P2P Connections
def reward_peer_connections(peers, resource_usage, network_state):
    """
    Reward function for P2P connections:
    - Rewards more connections during low load (for redundancy)
    - Rewards fewer connections during high load (to save resources)
    - Penalizes very high connections when resources are constrained
    """
    # Initialize reward
    reward = 0
    
    # Check if within reasonable range
    if 40 <= peers <= 70:
        reward += 3
    else:
        reward -= 5
    
    # Network state specific rewards
    if network_state == NetworkState.LOW and resource_usage < 70:
        # In low congestion with available resources, reward more peers
        normalized_peers = (peers - 40) / (70 - 40)  # 0-1 scale
        reward += normalized_peers * 4
    elif network_state == NetworkState.HIGH or resource_usage > 80:
        # In high congestion or high resource usage, reward fewer peers
        normalized_peers = 1 - (peers - 40) / (70 - 40)  # 1-0 scale
        reward += normalized_peers * 4
    
    return reward

# ✅ Reward Function for Block Size Adjustment
def reward_block_size(block_size_adj, network_state, latency):
    """
    Reward function for block size adjustment:
    - Rewards positive adjustments during low congestion
    - Rewards negative adjustments during high congestion
    - Penalizes extreme adjustments
    """
    # Initialize reward
    reward = 0
    
    # Penalize extreme adjustments
    if abs(block_size_adj) > 8:
        reward -= 3
    
    # Network state specific rewards
    if network_state == NetworkState.LOW and latency < 1000:
        # In low congestion with good latency, reward positive adjustments
        if block_size_adj > 0:
            reward += block_size_adj / 2  # Scale reward with adjustment size
    elif network_state == NetworkState.HIGH or latency > 1500:
        # In high congestion or high latency, reward negative adjustments
        if block_size_adj < 0:
            reward += abs(block_size_adj) / 2
    
    return reward

# ✅ Simulate Network Data
def simulate_network_data(batch_size=32):
    """
    Simulate batch of network data for training
    """
    latency = np.random.randint(100, 2000, batch_size)  # 100-2000ms latency
    uptime = np.random.randint(80, 100, batch_size)  # 80-100% uptime
    resource_usage = np.random.randint(30, 95, batch_size)  # 30-95% CPU/memory usage
    throughput = np.random.randint(500, 2000, batch_size)  # 500-2000 TPS
    pending_tx_count = np.random.randint(10, 500, batch_size)  # 10-500 pending transactions
    block_utilization = np.random.randint(10, 95, batch_size)  # 10-95% block utilization
    mempool_size = np.random.randint(50, 1000, batch_size)  # 50-1000 mempool size
    
    # Create target values based on domain knowledge
    gas_limit_targets = np.zeros(batch_size)
    fees_targets = np.zeros((batch_size, 3))  # [slow, medium, fast]
    peer_connections_targets = np.zeros(batch_size)
    block_size_adj_targets = np.zeros(batch_size)
    
    # Set target values based on network conditions
    for i in range(batch_size):
        network_state = determine_network_state(pending_tx_count[i], block_utilization[i])
        
        # Gas limit targets
        if network_state == NetworkState.LOW:
            gas_limit_targets[i] = np.random.randint(32000000, 35000000)
        elif network_state == NetworkState.MEDIUM:
            gas_limit_targets[i] = np.random.randint(30000000, 32000000)
        else:  # HIGH
            gas_limit_targets[i] = np.random.randint(28000000, 30000000)
        
        # Fee targets - base values
        base_fee = 5 if network_state == NetworkState.LOW else \
                  10 if network_state == NetworkState.MEDIUM else 20
        
        fees_targets[i, 0] = base_fee + np.random.randint(0, 3)  # Slow
        fees_targets[i, 1] = base_fee * 1.5 + np.random.randint(0, 5)  # Medium
        fees_targets[i, 2] = base_fee * 2.5 + np.random.randint(0, 8)  # Fast
        
        # Peer connection targets
        if network_state == NetworkState.LOW and resource_usage[i] < 70:
            peer_connections_targets[i] = np.random.randint(60, 70)  # More peers during low load
        elif network_state == NetworkState.HIGH or resource_usage[i] > 80:
            peer_connections_targets[i] = np.random.randint(40, 50)  # Fewer peers during high load
        else:
            peer_connections_targets[i] = np.random.randint(50, 60)  # Medium peers
        
        # Block size adjustment targets
        if network_state == NetworkState.LOW and latency[i] < 1000:
            block_size_adj_targets[i] = np.random.uniform(2, 10)  # Increase during low congestion
        elif network_state == NetworkState.HIGH or latency[i] > 1500:
            block_size_adj_targets[i] = np.random.uniform(-10, -2)  # Decrease during high congestion
        else:
            block_size_adj_targets[i] = np.random.uniform(-2, 2)  # Small adjustments for medium
    
    # Normalize input data
    input_data = np.column_stack([
        latency / 2000,
        uptime / 100,
        resource_usage / 100,
        throughput / 2000,
        pending_tx_count / 500,
        block_utilization / 100,
        mempool_size / 1000
    ])
    
    # Convert to PyTorch tensors
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    gas_limit_tensor = torch.tensor(gas_limit_targets, dtype=torch.float32)
    fees_tensor = torch.tensor(fees_targets, dtype=torch.float32)
    peer_connections_tensor = torch.tensor(peer_connections_targets, dtype=torch.float32)
    block_size_adj_tensor = torch.tensor(block_size_adj_targets, dtype=torch.float32)
    
    # Return raw data and tensors for model training
    return {
        'raw_data': {
            'latency': latency,
            'uptime': uptime,
            'resource_usage': resource_usage,
            'throughput': throughput,
            'pending_tx_count': pending_tx_count,
            'block_utilization': block_utilization,
            'mempool_size': mempool_size
        },
        'tensors': {
            'input': input_tensor,
            'gas_limit': gas_limit_tensor / 35000000,  # Normalize to 0-1
            'fees': fees_tensor / 50,  # Normalize to 0-1
            'peer_connections': (peer_connections_tensor - 40) / 30,  # Normalize to 0-1
            'block_size_adj': block_size_adj_tensor / 100  # Convert to -0.1 to 0.1
        }
    }

# ✅ Train the AI Model with Enhanced Reward System
def train_ai(epochs=1000, batch_size=32, lr=0.001):
    print("🚀 Starting Blockchain Optimizer AI Training...")
    
    # Initialize model
    model = BlockchainOptimizerAI()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Define loss functions
    gas_limit_criterion = nn.MSELoss()
    fees_criterion = nn.MSELoss()
    peer_connections_criterion = nn.MSELoss()
    block_size_criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(epochs):
        # Generate batch of simulated data
        data = simulate_network_data(batch_size)
        inputs = data['tensors']['input']
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Calculate individual losses
        gas_limit_loss = gas_limit_criterion(outputs['gas_limit'], 
                                           data['tensors']['gas_limit'].unsqueeze(1))
        
        fees_loss = fees_criterion(outputs['fees'], 
                                 data['tensors']['fees'])
        
        peer_connections_loss = peer_connections_criterion(outputs['peer_connections'], 
                                                        data['tensors']['peer_connections'].unsqueeze(1))
        
        block_size_loss = block_size_criterion(outputs['block_size_adjustment'], 
                                             data['tensors']['block_size_adj'].unsqueeze(1))
        
        # Total loss is weighted sum of individual losses
        total_loss = gas_limit_loss * 0.4 + fees_loss * 0.3 + \
                    peer_connections_loss * 0.2 + block_size_loss * 0.1
        
        # Calculate rewards for each prediction
        total_reward = 0
        raw_data = data['raw_data']
        
        for i in range(batch_size):
            # Get network state for this sample
            network_state = determine_network_state(
                raw_data['pending_tx_count'][i], 
                raw_data['block_utilization'][i]
            )
            
            # Calculate rewards for each prediction
            gas_limit_reward = reward_gas_limit(
                outputs['gas_limit'][i].item() * 35000000,  # Denormalize
                raw_data['pending_tx_count'][i],
                raw_data['block_utilization'][i]
            )
            
            fees_reward = reward_fee_estimation(
                outputs['fees'][i].detach().numpy() * 50,  # Denormalize
                raw_data['pending_tx_count'][i],
                raw_data['block_utilization'][i]
            )
            
            peer_connections_reward = reward_peer_connections(
                outputs['peer_connections'][i].item() * 30 + 40,  # Denormalize
                raw_data['resource_usage'][i],
                network_state
            )
            
            block_size_reward = reward_block_size(
                outputs['block_size_adjustment'][i].item() * 100,  # Convert to percentage
                network_state,
                raw_data['latency'][i]
            )
            
            # Sum rewards (weighted by importance)
            sample_reward = (gas_limit_reward * 0.4 + 
                           fees_reward * 0.3 + 
                           peer_connections_reward * 0.2 + 
                           block_size_reward * 0.1)
            
            total_reward += sample_reward
        
        # Average reward for the batch
        avg_reward = total_reward / batch_size
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={total_loss.item():.4f}, Reward={avg_reward:.4f}")
    
    # ✅ Save trained model
    save_path = "ai_model/blockchain_optimizer_model.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved successfully at {save_path}")
    
    # Return the trained model
    return model

# ✅ Run Training Script
if __name__ == "__main__":
    # Train the model with 1000 epochs
    model = train_ai(epochs=1000, batch_size=64, lr=0.001)
    
    # Show example prediction after training
    test_input = torch.tensor([[
        500/2000,     # latency
        98/100,       # uptime
        70/100,       # resource usage
        1000/2000,    # throughput
        150/500,      # pending tx count
        65/100,       # block utilization
        200/1000      # mempool size
    ]], dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        predictions = model(test_input)
    
    print("\n📊 Example Prediction After Training:")
    print(f"Gas Limit: {predictions['gas_limit'].item() * 35000000:.0f}")
    print(f"Fees (Gwei): {[round(f.item() * 50, 2) for f in predictions['fees'][0]]}")
    print(f"Peer Connections: {predictions['peer_connections'].item() * 30 + 40:.0f}")
    print(f"Block Size Adjustment: {predictions['block_size_adjustment'].item() * 100:.2f}%")