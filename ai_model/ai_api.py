from flask import Flask, request, jsonify
import torch
from web3 import Web3
from web3.middleware import geth_poa_middleware
from flask_cors import CORS
from ai_model import predict_gas_limit
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Ethereum Configuration
INFURA_URL = "https://sepolia.infura.io/v3/b5f2559745fc404a8b630669565301b1"
ACCOUNT_ADDRESS = "0x8a5F3239b4A9142b173B421BD940e72f69BEB213"
PRIVATE_KEY = "11d740c6a4ba765257b909abe790916998b4ed200467ced9e707ef560be42049"
CONTRACT_ADDRESS = "0x0Eab42a7c6262B0be833Ed4C2dC0070faEa480EF"

# Initialize Web3
def initialize_web3():
    web3 = Web3(Web3.HTTPProvider(INFURA_URL))
    web3.middleware_onion.inject(geth_poa_middleware, layer=0)
    if not web3.is_connected():
        raise Exception("Failed to connect to Ethereum network")
    return web3

# Contract ABI
CONTRACT_ABI = [
    {
        "inputs": [
            {"internalType": "uint256", "name": "_initialGasLimit", "type": "uint256"},
            {"internalType": "uint256", "name": "_maxGasLimit", "type": "uint256"}
        ],
        "stateMutability": "nonpayable",
        "type": "constructor"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": False, "internalType": "uint256", "name": "newLimit", "type": "uint256"}
        ],
        "name": "GasLimitUpdated",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": False, "internalType": "address", "name": "nodeAddress", "type": "address"}
        ],
        "name": "NodeRegistered",
        "type": "event"
    },
    {
        "inputs": [],
        "name": "currentGasLimit",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "maxGasLimit",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "uint256", "name": "newLimit", "type": "uint256"}],
        "name": "setGasLimit",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

# Initialize web3 and contract
try:
    web3 = initialize_web3()
    contract = web3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)
    logger.info("Web3 and contract initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Web3: {str(e)}")
    raise

def verify_contract():
    """Verify contract is accessible and functional"""
    try:
        code = web3.eth.get_code(CONTRACT_ADDRESS)
        if code == b'':
            raise Exception("No contract found at specified address")
        
        gas_limit = contract.functions.currentGasLimit().call()
        logger.info(f"Contract verified. Current gas limit: {gas_limit}")
        return True
    except Exception as e:
        logger.error(f"Contract verification failed: {str(e)}")
        return False

def get_blockchain_data():
    """Get current blockchain metrics"""
    try:
        latest_block = web3.eth.get_block('latest')
        pending_txs = web3.eth.get_block_transaction_count(latest_block.number)
        gas_used = latest_block.gasUsed
        gas_limit = latest_block.gasLimit
        
        logger.info(f"Blockchain data retrieved - Pending TXs: {pending_txs}, Gas Used: {gas_used}")
        return pending_txs, gas_used, gas_limit
    except Exception as e:
        logger.error(f"Failed to get blockchain data: {str(e)}")
        raise

@app.route("/predict_gas", methods=["POST"])
def predict_gas():
    """Endpoint to predict optimal gas limit"""
    try:
        pending_txs, gas_used, gas_limit = get_blockchain_data()
        predicted_gas_limit = predict_gas_limit(pending_txs, gas_used, gas_limit)
        
        return jsonify({
            "success": True,
            "new_gas_limit": predicted_gas_limit,
            "current_gas_limit": gas_limit,
            "metrics": {
                "pending_transactions": pending_txs,
                "current_gas_used": gas_used
            }
        })
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/update_gas", methods=["POST"])
def update_gas():
    """Endpoint to update gas limit in smart contract"""
    try:
        data = request.get_json()
        new_gas_limit = data["new_gas_limit"]
        
        # Validate gas limit
        current_limit = contract.functions.currentGasLimit().call()
        max_limit = contract.functions.maxGasLimit().call()
        
        if new_gas_limit > max_limit:
            raise ValueError(f"Gas limit exceeds maximum allowed: {max_limit}")
        
        # Prepare transaction
        nonce = web3.eth.get_transaction_count(ACCOUNT_ADDRESS)
        gas_price = web3.eth.gas_price
        
        tx = contract.functions.setGasLimit(new_gas_limit).build_transaction({
            'from': ACCOUNT_ADDRESS,
            'nonce': nonce,
            'gas': 200000,  # Increased gas limit for safety
            'gasPrice': gas_price
        })
        
        # Sign and send transaction
        signed_tx = web3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        logger.info(f"Gas limit update transaction sent: {tx_hash.hex()}")
        
        return jsonify({
            "success": True,
            "tx_hash": tx_hash.hex(),
            "updated_gas_limit": new_gas_limit,
            "previous_gas_limit": current_limit
        })

    except Exception as e:
        logger.error(f"Update failed: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Verify contract on startup
if not verify_contract():
    raise Exception("Contract verification failed during startup")

if __name__ == "_main_":
    app.run(port=5000, debug=True)