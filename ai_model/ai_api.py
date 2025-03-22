from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from web3 import Web3
from web3.middleware import geth_poa_middleware
import os
import logging
import time
from dotenv import load_dotenv
from ai_model import get_blockchain_optimization

# ✅ Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Load environment variables from .env file
load_dotenv()

# ✅ Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# ✅ Connect to Ethereum via Infura
INFURA_URL = os.getenv("INFURA_URL", "https://sepolia.infura.io/v3/b5f2559745fc404a8b630669565301b1")
web3 = Web3(Web3.HTTPProvider(INFURA_URL))

# ✅ Add PoA middleware (Required for Sepolia, Polygon, etc.)
web3.middleware_onion.inject(geth_poa_middleware, layer=0)

# ✅ Load Wallet & Private Key from environment variables
ACCOUNT_ADDRESS = os.getenv("ACCOUNT_ADDRESS", "0x8a5F3239b4A9142b173B421BD940e72f69BEB213")
PRIVATE_KEY = os.getenv("PRIVATE_KEY", "11d740c6a4ba765257b909abe790916998b4ed200467ced9e707ef560be42049")

# ✅ Load Smart Contract
contract_address = os.getenv("CONTRACT_ADDRESS", "0xe14bac79c6b465b3c201cbd4f86693ae24cac06c")
contract_abi = [
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_initialGasLimit",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "_maxGasLimit",
          "type": "uint256"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "constructor"
    },
    {
      "anonymous": False,
      "inputs": [
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "newLimit",
          "type": "uint256"
        }
      ],
      "name": "GasLimitUpdated",
      "type": "event"
    },
    {
      "anonymous": False,
      "inputs": [
        {
          "indexed": False,
          "internalType": "address",
          "name": "nodeAddress",
          "type": "address"
        }
      ],
      "name": "NodeRegistered",
      "type": "event"
    },
    {
      "inputs": [],
      "name": "currentGasLimit",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "getRegisteredNodes",
      "outputs": [
        {
          "internalType": "address[]",
          "name": "",
          "type": "address[]"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "maxGasLimit",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "",
          "type": "address"
        }
      ],
      "name": "nodes",
      "outputs": [
        {
          "internalType": "address",
          "name": "nodeAddress",
          "type": "address"
        },
        {
          "internalType": "bool",
          "name": "isActive",
          "type": "bool"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "owner",
      "outputs": [
        {
          "internalType": "address",
          "name": "",
          "type": "address"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "nodeAddress",
          "type": "address"
        }
      ],
      "name": "registerNode",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "name": "registeredNodes",
      "outputs": [
        {
          "internalType": "address",
          "name": "",
          "type": "address"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "newLimit",
          "type": "uint256"
        }
      ],
      "name": "setGasLimit",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    }
]

try:
    contract = web3.eth.contract(address=contract_address, abi=contract_abi)
except Exception as e:
    logger.error(f"Error initializing contract: {e}")

# ✅ Get blockchain data
def get_blockchain_data():
    """Get current blockchain metrics"""
    try:
        latest_block = web3.eth.get_block('latest')
        pending_txs = web3.eth.get_block_transaction_count('pending')
        gas_used = latest_block.gasUsed
        gas_limit = latest_block.gasLimit
        
        # Get transaction activity for the last few blocks to estimate TPS
        block_count = 10
        total_tx_count = 0
        end_block = latest_block.number
        
        for i in range(block_count):
            block_num = end_block - i
            if block_num >= 0:
                block = web3.eth.get_block(block_num)
                total_tx_count += len(block.transactions)
        
        # Calculate transactions per second (rough estimate)
        # Assuming ~13 seconds per block on Ethereum
        tps = total_tx_count / (block_count * 13)
        
        # Calculate block utilization
        block_utilization = (gas_used / gas_limit) * 100
        
        return {
            'pending_txs': pending_txs,
            'gas_used': gas_used,
            'gas_limit': gas_limit,
            'block_utilization': block_utilization,
            'tps': tps,
            'latest_block': latest_block.number
        }
    except Exception as e:
        logger.warning(f"Error fetching blockchain data: {e}")
        # Return mock data if blockchain access fails
        return {
            'pending_txs': 150,
            'gas_used': 7000000,
            'gas_limit': 15000000,
            'block_utilization': 65,
            'tps': 15,
            'latest_block': 0
        }

# ✅ Get mock node metrics
def get_node_metrics():
    """Get node performance metrics (mocked for demo)"""
    return {
        'latency': 500 + int(200 * (0.5 - (time.time() % 10) / 10)),  # 300-700ms varying with time
        'uptime': 98.5,
        'resource_usage': 70 + int(20 * (0.5 - (time.time() % 20) / 20)),  # 50-90% varying with time
        'throughput': 1000 + int(500 * (0.5 - (time.time() % 15) / 15))  # 500-1500 TPS varying with time
    }

# ✅ AI Predicts Comprehensive Blockchain Optimization
@app.route("/predict_gas", methods=["POST"])
def predict_gas():
    """Endpoint to get AI-powered blockchain optimization recommendations"""
    try:
        # Get data from request or use mock/default values
        data = request.get_json() or {}
        
        # Get node metrics (either from request or mocked)
        node_metrics = {
            'latency': data.get('latency', get_node_metrics()['latency']),
            'uptime': data.get('uptime', get_node_metrics()['uptime']),
            'resource_usage': data.get('resource_usage', get_node_metrics()['resource_usage']),
            'throughput': data.get('throughput', get_node_metrics()['throughput'])
        }
        
        # Get blockchain data (either from actual blockchain or mocked)
        try:
            blockchain_data = get_blockchain_data()
            pending_tx_count = blockchain_data['pending_txs']
            block_utilization = blockchain_data['block_utilization']
            mempool_size = pending_tx_count  # Use pending txs as proxy for mempool size
        except Exception as e:
            logger.warning(f"Using default blockchain data: {e}")
            pending_tx_count = data.get('pending_tx_count', 150)
            block_utilization = data.get('block_utilization', 65)
            mempool_size = data.get('mempool_size', 200)
        
        # Get AI optimization recommendations
        optimization = get_blockchain_optimization(
            latency=node_metrics['latency'],
            uptime=node_metrics['uptime'],
            resource_usage=node_metrics['resource_usage'],
            throughput=node_metrics['throughput'],
            pending_tx_count=pending_tx_count,
            block_utilization=block_utilization,
            mempool_size=mempool_size
        )
        
        # Add additional blockchain metrics if available
        if 'tps' in blockchain_data:
            optimization['metrics']['tps'] = blockchain_data['tps']
        if 'latest_block' in blockchain_data:
            optimization['metrics']['latest_block'] = blockchain_data['latest_block']
        
        # Return comprehensive optimization data
        return jsonify({
            "success": True,
            "optimization": optimization
        })
    
    except Exception as e:
        logger.error(f"Error in predict_gas: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "fallback_gas_limit": 15000000  # Fallback value
        })

# ✅ AI Updates Smart Contract Gas Limit
@app.route("/update_gas", methods=["POST"])
def update_gas():
    """Endpoint to update gas limit on the blockchain"""
    try:
        data = request.get_json()
        new_gas_limit = data.get("new_gas_limit")
        
        if not new_gas_limit:
            return jsonify({"success": False, "error": "new_gas_limit is required"}), 400
        
        # Get current gas price for optimal transaction execution
        gas_price = web3.eth.gas_price
        
        # Add 10% to the gas price to ensure quick confirmation
        adjusted_gas_price = int(gas_price * 1.1)
        
        # Get nonce for sender account
        nonce = web3.eth.get_transaction_count(ACCOUNT_ADDRESS)
        
        # Build transaction to update gas limit
        tx = contract.functions.setGasLimit(int(new_gas_limit)).build_transaction({
            'from': ACCOUNT_ADDRESS,
            'nonce': nonce,
            'gas': 200000,  # Fixed gas limit for the transaction itself
            'gasPrice': adjusted_gas_price
        })
        
        # Sign transaction with private key
        signed_tx = web3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        
        # Send transaction
        tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)
        
        # Return transaction hash
        return jsonify({
            "success": True, 
            "tx_hash": tx_hash.hex(), 
            "message": f"Gas limit update to {new_gas_limit} submitted to the blockchain"
        })
        
    except Exception as e:
        logger.error(f"Error updating gas limit: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ✅ Get Network Metrics
@app.route("/network_metrics", methods=["GET"])
def network_metrics():
    """Endpoint to get current network metrics"""
    try:
        # Get blockchain data
        blockchain_data = get_blockchain_data()
        
        # Get node metrics
        node_metrics = get_node_metrics()
        
        # Combine data
        metrics = {
            "blockchain": blockchain_data,
            "node": node_metrics,
            "timestamp": int(time.time())
        }
        
        return jsonify({
            "success": True,
            "metrics": metrics
        })
        
    except Exception as e:
        logger.error(f"Error getting network metrics: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ✅ Ebpf Traffic Control Rules
@app.route("/optimize_traffic", methods=["POST"])
def optimize_traffic():
    """Endpoint to get eBPF traffic control rules based on network state"""
    try:
        data = request.get_json() or {}
        
        # Get blockchain metrics
        blockchain_data = get_blockchain_data()
        
        # Determine traffic rules based on congestion
        block_utilization = blockchain_data['block_utilization']
        pending_txs = blockchain_data['pending_txs']
        
        # Default rules (medium congestion)
        traffic_rules = {
            "priority_accounts": [],
            "drop_patterns": [],
            "batch_threshold": 5,  # Batch every 5 similar transactions
            "max_txs_per_sender": 20
        }
        
        # Adjust rules based on congestion
        if block_utilization < 30 and pending_txs < 100:
            # Low congestion - Relaxed rules
            traffic_rules["batch_threshold"] = 10
            traffic_rules["max_txs_per_sender"] = 50
        elif block_utilization > 70 or pending_txs > 300:
            # High congestion - Strict rules
            traffic_rules["batch_threshold"] = 3
            traffic_rules["max_txs_per_sender"] = 10
            traffic_rules["drop_patterns"] = [
                {"pattern": "spam_", "description": "Known spam prefix"},
                {"pattern": "0x000000", "description": "Low-value zero tx"}
            ]
            traffic_rules["priority_accounts"] = [
                {"address": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D", "description": "Uniswap Router"}
            ]
        
        return jsonify({
            "success": True,
            "traffic_rules": traffic_rules,
            "network_state": "High" if block_utilization > 70 else "Low" if block_utilization < 30 else "Medium"
        })
        
    except Exception as e:
        logger.error(f"Error optimizing traffic: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ✅ Start Flask API
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)