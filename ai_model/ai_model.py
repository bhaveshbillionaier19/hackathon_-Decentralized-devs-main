import os
import torch
import torch.nn as nn
import numpy as np
from enum import Enum

# ✅ Network Congestion State Enum
class NetworkState(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2

# ✅ Advanced AI Model for Blockchain Optimization
class BlockchainOptimizerAI(nn.Module):
    def __init__(self):
        super(BlockchainOptimizerAI, self).__init__()
        # Input features: 
        # - Latency, Uptime, Resource Usage, Throughput, 
        # - Pending TX Count, Current Block Utilization, Mempool Size
        self.input_size = 7
        
        # Hidden layers with increasing complexity
        self.fc1 = nn.Linear(self.input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Output layers for different parameters
        self.gas_limit_layer = nn.Linear(64, 1)  # Gas limit prediction
        self.fee_estimator_layer = nn.Linear(64, 3)  # Fee estimation (slow, medium, fast)
        self.peer_connections_layer = nn.Linear(64, 1)  # Optimal peer connections
        self.block_size_layer = nn.Linear(64, 1)  # Optimal block size adjustment

    def forward(self, x):
        # Shared feature extraction layers
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        
        # Output specific predictions
        gas_limit = torch.sigmoid(self.gas_limit_layer(x)) * 35000000  # Scale to max gas limit
        
        # Fee estimation (in Gwei)
        # Output [slow_fee, medium_fee, fast_fee] relative to base fee
        fees = torch.sigmoid(self.fee_estimator_layer(x)) * 50  # Scale to max of 50 Gwei
        
        # Peer connection optimization (40-70 range)
        peer_connections = torch.sigmoid(self.peer_connections_layer(x)) * 30 + 40
        
        # Block size adjustment (-10% to +10%)
        block_size_adjustment = torch.tanh(self.block_size_layer(x)) * 0.1  # -10% to +10%
        
        return {
            'gas_limit': gas_limit,
            'fees': fees,
            'peer_connections': peer_connections,
            'block_size_adjustment': block_size_adjustment
        }

# ✅ Load trained model
def load_model():
    model = BlockchainOptimizerAI()

    # 🔹 Check if model file exists before loading
    model_path = "ai_model/blockchain_optimizer_model.pth"
    if not os.path.exists(model_path):
        print("🚨 Warning: Model file not found! Using untrained model.")
        return model

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# ✅ Determine Network Congestion State
def determine_network_state(pending_tx_count, block_utilization):
    if block_utilization < 30 and pending_tx_count < 50:
        return NetworkState.LOW
    elif block_utilization < 70 and pending_tx_count < 200:
        return NetworkState.MEDIUM
    else:
        return NetworkState.HIGH

# ✅ Predict Optimal Gas Limit
def predict_gas_limit(latency, uptime, resource_usage, throughput, 
                     pending_tx_count=100, block_utilization=50, mempool_size=100):
    """
    Predict optimal gas limit based on network conditions
    
    Args:
        latency (float): Network latency in ms
        uptime (float): Node uptime percentage
        resource_usage (float): CPU/Memory resource usage percentage
        throughput (float): Current transactions throughput
        pending_tx_count (int): Number of pending transactions
        block_utilization (float): Current block utilization percentage
        mempool_size (int): Current mempool size
        
    Returns:
        int: Optimal gas limit
    """
    model = load_model()
    
    # Determine if we're using an untrained model
    model_path = "ai_model/blockchain_optimizer_model.pth"
    using_untrained_model = not os.path.exists(model_path)
    
    if using_untrained_model:
        # If using untrained model, use more dynamic calculations based on inputs directly
        
        # Base gas limit value
        base_gas_limit = 30000000
        
        # Adjust based on network congestion
        network_state = determine_network_state(pending_tx_count, block_utilization)
        
        # Calculate dynamic gas limit based on network conditions
        if network_state == NetworkState.LOW:
            # Low congestion - higher gas limit
            congestion_factor = 1.1
        elif network_state == NetworkState.HIGH:
            # High congestion - lower gas limit
            congestion_factor = 0.9
        else:
            # Medium congestion - baseline
            congestion_factor = 1.0
            
        # Factor in resource usage
        resource_factor = 1.0 - (resource_usage - 50) / 100
        
        # Calculate final gas limit
        gas_limit = int(base_gas_limit * congestion_factor * resource_factor)
        
        # Ensure gas limit is within reasonable bounds
        return max(28000000, min(gas_limit, 35000000))
        
    # Use AI model if available
    # Normalize inputs
    input_data = torch.tensor([
        [
            latency / 2000,  # Normalize latency (0-2000ms)
            uptime / 100,    # Normalize uptime (0-100%)
            resource_usage / 100,  # Normalize resource usage (0-100%)
            throughput / 2000,  # Normalize throughput (0-2000 TPS)
            pending_tx_count / 500,  # Normalize pending tx count
            block_utilization / 100,  # Normalize block utilization
            mempool_size / 1000  # Normalize mempool size
        ]
    ], dtype=torch.float32)
    
    # Get AI predictions
    with torch.no_grad():
        predictions = model(input_data)
    
    # Get the raw gas limit prediction
    gas_limit_prediction = int(predictions['gas_limit'].item())
    
    # Apply network state adjustments
    network_state = determine_network_state(pending_tx_count, block_utilization)
    
    if network_state == NetworkState.LOW:
        # During low congestion, increase gas limit for better throughput
        adjusted_gas_limit = int(gas_limit_prediction * 1.1)  # +10%
    elif network_state == NetworkState.HIGH:
        # During high congestion, decrease gas limit for stability
        adjusted_gas_limit = int(gas_limit_prediction * 0.9)  # -10%
    else:
        # Medium congestion - use AI prediction directly
        adjusted_gas_limit = gas_limit_prediction
    
    # Ensure gas limit is within reasonable bounds (28M to 35M)
    return max(28000000, min(adjusted_gas_limit, 35000000))

# ✅ Estimate Optimal Gas Fees
def predict_gas_fees(latency, uptime, resource_usage, throughput,
                    pending_tx_count=100, block_utilization=50, mempool_size=100):
    """
    Predict optimal gas fees based on network conditions
    
    Returns:
        dict: Estimated gas fees for slow, medium, and fast transactions
    """
    model = load_model()
    
    # Determine if we're using an untrained model
    model_path = "ai_model/blockchain_optimizer_model.pth"
    using_untrained_model = not os.path.exists(model_path)
    
    # Set time estimates based on network congestion
    network_state = determine_network_state(pending_tx_count, block_utilization)
    
    if network_state == NetworkState.LOW:
        time_estimates = {
            'slow': '1-3 min',
            'medium': '30-60 sec',
            'fast': '<30 sec'
        }
    elif network_state == NetworkState.MEDIUM:
        time_estimates = {
            'slow': '5-10 min',
            'medium': '1-3 min',
            'fast': '30-60 sec'
        }
    else:  # HIGH congestion
        time_estimates = {
            'slow': '10-20 min',
            'medium': '5-10 min',
            'fast': '1-3 min'
        }
    
    if using_untrained_model:
        # If using untrained model, calculate dynamic fees based on inputs
        
        # Base fee values depend on network congestion
        if network_state == NetworkState.LOW:
            base_fee = 5
        elif network_state == NetworkState.MEDIUM:
            base_fee = 15
        else:  # HIGH
            base_fee = 30
            
        # Adjust based on pending transactions and block utilization
        congestion_multiplier = 1.0 + (pending_tx_count / 1000) + (block_utilization / 200)
        
        # Calculate fees for each speed level
        slow_fee = max(1, round(base_fee * 0.7 * congestion_multiplier, 2))
        medium_fee = max(2, round(base_fee * 1.0 * congestion_multiplier, 2))
        fast_fee = max(3, round(base_fee * 1.5 * congestion_multiplier, 2))
        
        return {
            'slow': {'fee': slow_fee, 'time': time_estimates['slow']},
            'medium': {'fee': medium_fee, 'time': time_estimates['medium']},
            'fast': {'fee': fast_fee, 'time': time_estimates['fast']},
        }
    
    # Use AI model if available
    # Normalize inputs
    input_data = torch.tensor([
        [
            latency / 2000,  # Normalize latency (0-2000ms)
            uptime / 100,    # Normalize uptime (0-100%)
            resource_usage / 100,  # Normalize resource usage (0-100%)
            throughput / 2000,  # Normalize throughput (0-2000 TPS)
            pending_tx_count / 500,  # Normalize pending tx count
            block_utilization / 100,  # Normalize block utilization
            mempool_size / 1000  # Normalize mempool size
        ]
    ], dtype=torch.float32)
    
    # Get AI predictions
    with torch.no_grad():
        predictions = model(input_data)
    
    # Get fee predictions
    fees = predictions['fees'][0]
    
    return {
        'slow': {'fee': round(fees[0].item(), 2), 'time': time_estimates['slow']},
        'medium': {'fee': round(fees[1].item(), 2), 'time': time_estimates['medium']},
        'fast': {'fee': round(fees[2].item(), 2), 'time': time_estimates['fast']},
    }

# ✅ Predict Optimal P2P Connections
def predict_peer_connections(latency, uptime, resource_usage, throughput,
                           pending_tx_count=100, block_utilization=50, mempool_size=100):
    """
    Predict optimal peer connections based on network conditions
    
    Returns:
        int: Optimal number of peer connections
    """
    model = load_model()
    
    # Normalize inputs
    input_data = torch.tensor([
        [
            latency / 2000,
            uptime / 100,
            resource_usage / 100,
            throughput / 2000,
            pending_tx_count / 500,
            block_utilization / 100,
            mempool_size / 1000
        ]
    ], dtype=torch.float32)
    
    # Get AI predictions
    with torch.no_grad():
        predictions = model(input_data)
    
    # Get peer connections prediction
    peer_connections = int(predictions['peer_connections'].item())
    
    # Apply network state adjustments
    network_state = determine_network_state(pending_tx_count, block_utilization)
    
    if network_state == NetworkState.LOW:
        # During low load, increase peers for better redundancy
        adjusted_peers = peer_connections + 10
    elif network_state == NetworkState.HIGH:
        # During high load, reduce peers to save resources
        adjusted_peers = max(40, peer_connections - 10)
    else:
        # Medium load - use AI prediction directly
        adjusted_peers = peer_connections
    
    # Ensure peer count is within reasonable bounds (40-70)
    return max(40, min(adjusted_peers, 70))

# ✅ Predict Optimal Block Size Adjustment
def predict_block_size_adjustment(latency, uptime, resource_usage, throughput,
                                pending_tx_count=100, block_utilization=50, mempool_size=100):
    """
    Predict optimal block size adjustment percentage based on network conditions
    
    Returns:
        float: Block size adjustment percentage (-10% to +10%)
    """
    model = load_model()
    
    # Normalize inputs
    input_data = torch.tensor([
        [
            latency / 2000,
            uptime / 100,
            resource_usage / 100,
            throughput / 2000,
            pending_tx_count / 500,
            block_utilization / 100,
            mempool_size / 1000
        ]
    ], dtype=torch.float32)
    
    # Get AI predictions
    with torch.no_grad():
        predictions = model(input_data)
    
    # Get block size adjustment prediction
    block_size_adjustment = predictions['block_size_adjustment'].item() * 100  # Convert to percentage
    
    return round(block_size_adjustment, 2)  # Return as percentage with 2 decimal places

# ✅ All-in-one prediction function for frontend
def get_blockchain_optimization(latency, uptime, resource_usage, throughput,
                              pending_tx_count=100, block_utilization=50, mempool_size=100):
    """
    Get all optimization parameters in one call
    
    Returns:
        dict: All optimization parameters
    """
    # Determine network congestion state
    network_state = determine_network_state(pending_tx_count, block_utilization)
    congestion_level = "Low" if network_state == NetworkState.LOW else \
                      "Medium" if network_state == NetworkState.MEDIUM else "High"
    
    # Get all predictions
    gas_limit = predict_gas_limit(latency, uptime, resource_usage, throughput, 
                                pending_tx_count, block_utilization, mempool_size)
    
    gas_fees = predict_gas_fees(latency, uptime, resource_usage, throughput,
                              pending_tx_count, block_utilization, mempool_size)
    
    peer_connections = predict_peer_connections(latency, uptime, resource_usage, throughput,
                                             pending_tx_count, block_utilization, mempool_size)
    
    block_size_adjustment = predict_block_size_adjustment(latency, uptime, resource_usage, throughput,
                                                       pending_tx_count, block_utilization, mempool_size)
    
    # Generate recommendations based on network state
    recommendations = []
    
    if network_state == NetworkState.LOW:
        recommendations.append("Increase gas limit to improve throughput")
        recommendations.append("Increase peer connections for better redundancy")
        if block_size_adjustment > 0:
            recommendations.append("Consider increasing block size for more transactions per block")
    elif network_state == NetworkState.HIGH:
        recommendations.append("Decrease gas limit to maintain network stability")
        recommendations.append("Reduce peer connections to save node resources")
        recommendations.append("Prioritize high-value transactions in mempool")
        if block_size_adjustment < 0:
            recommendations.append("Consider decreasing block size for faster propagation")
    
    return {
        'current_gas_limit': gas_limit,
        'gas_fee_estimates': gas_fees,
        'optimal_peer_connections': peer_connections,
        'block_size_adjustment': block_size_adjustment,
        'network_congestion': congestion_level,
        'recommendations': recommendations,
        'metrics': {
            'pending_transactions': pending_tx_count,
            'current_gas_used': int(block_utilization * gas_limit / 100)
        }
    }

# ✅ Example Usage
if __name__ == "__main__":
    # Example for testing purposes
    optimization = get_blockchain_optimization(
        latency=500,         # 500ms latency
        uptime=98,           # 98% uptime
        resource_usage=70,   # 70% CPU/memory usage
        throughput=1000,     # 1000 TPS throughput
        pending_tx_count=150,  # 150 pending transactions
        block_utilization=65,  # 65% block utilization
        mempool_size=200     # 200 mempool size
    )
    
    print("Blockchain Optimization Results:")
    print(f"Gas Limit: {optimization['current_gas_limit']}")
    print(f"Gas Fees: {optimization['gas_fee_estimates']}")
    print(f"Peer Connections: {optimization['optimal_peer_connections']}")
    print(f"Block Size Adjustment: {optimization['block_size_adjustment']}%")
    print(f"Network Congestion: {optimization['network_congestion']}")
    print("Recommendations:")
    for rec in optimization['recommendations']:
        print(f"- {rec}")