# from flask import Flask, request, jsonify
# import torch
# from web3 import Web3
# from web3.middleware import geth_poa_middleware  # Works for latest Web3


# from ai_model import predict_gas_limit

# app = Flask(__name__)

# # 🔹 Connect to Ethereum via Infura
# INFURA_URL = "https://sepolia.infura.io/v3/b5f2559745fc404a8b630669565301b1"  # Replace with your Infura key
# web3 = Web3(Web3.HTTPProvider(INFURA_URL))

# # 🔹 Add PoA middleware (Required for Polygon, Binance Smart Chain, etc.)
# web3.middleware_onion.inject(geth_poa_middleware, layer=0)

# # 🔹 Manually set an Ethereum wallet (Replace with your actual details)
# ACCOUNT_ADDRESS = "0x8a5F3239b4A9142b173B421BD940e72f69BEB213"  # Your Ethereum wallet address
# PRIVATE_KEY = "11d740c6a4ba765257b909abe790916998b4ed200467ced9e707ef560be42049"  # ⚠️ Keep this secure!

# # 🔹 Load Smart Contract
# contract_address = "0x0Eab42a7c6262B0be833Ed4C2dC0070faEa480EF"  # Replace with actual contract address
# contract_abi =  [
#     {
#       "inputs": [
#         {
#           "internalType": "uint256",
#           "name": "_initialGasLimit",
#           "type": "uint256"
#         },
#         {
#           "internalType": "uint256",
#           "name": "_maxGasLimit",
#           "type": "uint256"
#         }
#       ],
#       "stateMutability": "nonpayable",
#       "type": "constructor"
#     },
#     {
#       "anonymous": ''False'',
#       "inputs": [
#         {
#           "indexed": ''False'',
#           "internalType": "uint256",
#           "name": "newLimit",
#           "type": "uint256"
#         }
#       ],
#       "name": "GasLimitUpdated",
#       "type": "event"
#     },
#     {
#       "anonymous": ''False'',
#       "inputs": [
#         {
#           "indexed": ''False'',
#           "internalType": "address",
#           "name": "nodeAddress",
#           "type": "address"
#         }
#       ],
#       "name": "NodeRegistered",
#       "type": "event"
#     },
#     {
#       "inputs": [],
#       "name": "currentGasLimit",
#       "outputs": [
#         {
#           "internalType": "uint256",
#           "name": "",
#           "type": "uint256"
#         }
#       ],
#       "stateMutability": "view",
#       "type": "function"
#     },
#     {
#       "inputs": [],
#       "name": "getRegisteredNodes",
#       "outputs": [
#         {
#           "internalType": "address[]",
#           "name": "",
#           "type": "address[]"
#         }
#       ],
#       "stateMutability": "view",
#       "type": "function"
#     },
#     {
#       "inputs": [],
#       "name": "maxGasLimit",
#       "outputs": [
#         {
#           "internalType": "uint256",
#           "name": "",
#           "type": "uint256"
#         }
#       ],
#       "stateMutability": "view",
#       "type": "function"
#     },
#     {
#       "inputs": [
#         {
#           "internalType": "address",
#           "name": "",
#           "type": "address"
#         }
#       ],
#       "name": "nodes",
#       "outputs": [
#         {
#           "internalType": "address",
#           "name": "nodeAddress",
#           "type": "address"
#         },
#         {
#           "internalType": "bool",
#           "name": "isActive",
#           "type": "bool"
#         }
#       ],
#       "stateMutability": "view",
#       "type": "function"
#     },
#     {
#       "inputs": [],
#       "name": "owner",
#       "outputs": [
#         {
#           "internalType": "address",
#           "name": "",
#           "type": "address"
#         }
#       ],
#       "stateMutability": "view",
#       "type": "function"
#     },
#     {
#       "inputs": [
#         {
#           "internalType": "address",
#           "name": "nodeAddress",
#           "type": "address"
#         }
#       ],
#       "name": "registerNode",
#       "outputs": [],
#       "stateMutability": "nonpayable",
#       "type": "function"
#     },
#     {
#       "inputs": [
#         {
#           "internalType": "uint256",
#           "name": "",
#           "type": "uint256"
#         }
#       ],
#       "name": "registeredNodes",
#       "outputs": [
#         {
#           "internalType": "address",
#           "name": "",
#           "type": "address"
#         }
#       ],
#       "stateMutability": "view",
#       "type": "function"
#     },
#     {
#       "inputs": [
#         {
#           "internalType": "uint256",
#           "name": "newLimit",
#           "type": "uint256"
#         }
#       ],
#       "name": "setGasLimit",
#       "outputs": [],
#       "stateMutability": "nonpayable",
#       "type": "function"
#     }
#   ]  # Add your contract ABI
# contract = web3.eth.contract(address=contract_address, abi=contract_abi)

# # 🔹 Get blockchain data (Pending TXs, Gas Used, Gas Limit)
# def get_blockchain_data():
#     latest_block = web3.eth.get_block('latest')
#     pending_txs = web3.eth.get_block_transaction_count(latest_block.number)
#     gas_used = latest_block.gasUsed
#     gas_limit = latest_block.gasLimit
#     return pending_txs, gas_used, gas_limit

# # ✅ AI Predicts Gas Limit
# @app.route("/predict_gas", methods=["POST"])
# def predict_gas():
#     pending_txs, gas_used, gas_limit = get_blockchain_data()
#     predicted_gas_limit = predict_gas_limit(pending_txs, gas_used, gas_limit)
#     return jsonify({"new_gas_limit": predicted_gas_limit})

# # ✅ AI Updates Smart Contract Gas Limit
# @app.route("/update_gas", methods=["POST"])
# def update_gas():
#     data = request.get_json()
#     new_gas_limit = data["new_gas_limit"]

#     try:
#         # 🔹 Get Nonce (Transaction count)
#         nonce = web3.eth.get_transaction_count(ACCOUNT_ADDRESS)

#         # 🔹 Build Transaction
#         tx = contract.functions.setGasLimit(new_gas_limit).build_transaction({
#             'from': ACCOUNT_ADDRESS,
#             'nonce': nonce,
#             'gas': 100000,
#             'gasPrice': web3.to_wei('50', 'gwei')  # Adjust gas price as needed
#         })

#         # 🔹 Sign Transaction with Private Key
#         signed_tx = web3.eth.account.sign_transaction(tx, PRIVATE_KEY)
#         # 🔹 Send Transaction
#         tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)
#         return jsonify({"tx_hash": tx_hash.hex(), "updated_gas_limit": new_gas_limit})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(port=5000)


from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Allow frontend to access API
import torch
from web3 import Web3
from web3.middleware import geth_poa_middleware
import os
from dotenv import load_dotenv
from ai_model import predict_gas_limit

# ✅ Load environment variables from .env file
load_dotenv()

# ✅ Initialize Flask app
app = Flask(__name__)
CORS(app)  # ✅ Allow React frontend to make requests

# ✅ Connect to Ethereum via Infura
INFURA_URL = os.getenv("INFURA_URL")
web3 = Web3(Web3.HTTPProvider(INFURA_URL))

# ✅ Add PoA middleware (Required for Sepolia, Polygon, etc.)
web3.middleware_onion.inject(geth_poa_middleware, layer=0)

# ✅ Load Wallet & Private Key from environment variables
ACCOUNT_ADDRESS = os.getenv("ACCOUNT_ADDRESS")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")

# ✅ Load Smart Contract
contract_address = os.getenv("CONTRACT_ADDRESS")
contract_abi =  [
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
  ]; # ✅ Replace with your actual ABI
contract = web3.eth.contract(address=contract_address, abi=contract_abi)

# ✅ AI Predicts Gas Limit
@app.route("/predict_gas", methods=["POST"])
def predict_gas():
    try:
        data = request.get_json()
        latency = data["latency"]
        uptime = data["uptime"]
        resource_usage = data["resource_usage"]
        throughput = data["throughput"]

        predicted_gas_limit = predict_gas_limit(latency, uptime, resource_usage, throughput)
        return jsonify({"new_gas_limit": predicted_gas_limit})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ AI Updates Smart Contract Gas Limit
@app.route("/update_gas", methods=["POST"])
def update_gas():
    try:
        data = request.get_json()
        new_gas_limit = data["new_gas_limit"]

        # Get Nonce (Transaction count)
        nonce = web3.eth.get_transaction_count(ACCOUNT_ADDRESS)

        # Build Transaction
        tx = contract.functions.setGasLimit(new_gas_limit).build_transaction({
            'from': ACCOUNT_ADDRESS,
            'nonce': nonce,
            'gas': 100000,
            'gasPrice': web3.to_wei('50', 'gwei')  # Adjust gas price if needed
        })

        # Sign Transaction with Private Key
        signed_tx = web3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        # Send Transaction
        tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)

        return jsonify({"tx_hash": tx_hash.hex(), "updated_gas_limit": new_gas_limit})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Start Flask API
if __name__ == "__main__":
    app.run(port=5000, debug=True)  # Debug mode enabled