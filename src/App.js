import React, { useState, useEffect } from "react";
import Web3 from "web3";

function App() {
  const [web3, setWeb3] = useState(null);
  const [account, setAccount] = useState("");
  const [contract, setContract] = useState(null);
  const [gasLimit, setGasLimit] = useState("");
  

  // Contract ABI
  const contractABI = 
  [
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
      "anonymous": false,
      "inputs": [
        {
          "indexed": false,
          "internalType": "uint256",
          "name": "newLimit",
          "type": "uint256"
        }
      ],
      "name": "GasLimitUpdated",
      "type": "event"
    },
    {
      "anonymous": false,
      "inputs": [
        {
          "indexed": false,
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
  ;

  const contractAddress = "0x0Eab42a7c6262B0be833Ed4C2dC0070faEa480EF"; // Replace with your contract address

  // Connect to MetaMask
  const connectMetaMask = async () => {
    if (window.ethereum) {
      try {
        const web3Instance = new Web3(window.ethereum);
        await window.ethereum.request({ method: "eth_requestAccounts" });
        const accounts = await web3Instance.eth.getAccounts();
        setWeb3(web3Instance);
        setAccount(accounts[0]);

        // Initialize contract
        const contractInstance = new web3Instance.eth.Contract(contractABI, contractAddress);
        setContract(contractInstance);
      } catch (error) {
        console.error("Error connecting to MetaMask:", error);
      }
    } else {
      alert("MetaMask not detected. Please install MetaMask.");
    }
  };

  // Fetch current gas limit
  const fetchGasLimit = async () => {
    if (contract) {
      
      
      const limit = await contract.methods.currentGasLimit().call();
      console.log(limit);
      setGasLimit(limit);
    }
  };

  // Update gas limit
  const updateGasLimit = async () => {
    if (contract) {
      const newLimit = gasLimit * 1.1; // Increase by 10%
      await contract.methods.setGasLimit(newLimit).send({ from: account });
      alert("Gas limit updated successfully!");
      fetchGasLimit(); // Refresh the displayed gas limit
    }
  };

  // Fetch gas limit on component mount
  useEffect(() => {
    if (contract) {
      fetchGasLimit();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [contract]);

  return (
    <div className="container mt-5">
      <h1 className="text-center">Autonomous Traffic Manager</h1>
      <div className="card p-4">
        {!web3 ? (
          <button className="btn btn-primary" onClick={connectMetaMask}>
            Connect MetaMask
          </button>
        ) : (
          <>
            <p>Connected Account: {account}</p>
            <p>Current Gas Limit: {gasLimit}</p>
            <button className="btn btn-success" onClick={updateGasLimit}>
              Update Gas Limit
            </button>
          </>
        )}
      </div>
    </div>
  );
}

export default App;