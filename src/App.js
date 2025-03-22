/* global BigInt */


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

  const contractAddress = "0xe14bac79c6b465b3c201cbd4f86693ae24cac06c"; // Replace with your contract address

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
    try {
      if (!contract) {
        console.error("Contract is not initialized");
        return;
      }
      
      console.log("Attempting to call currentGasLimit() on contract at:", contractAddress);
      
      // Try a different way to call the function
      const limit = await contract.methods.currentGasLimit().call({
        from: account,
        gas: 486600 // Explicitly set higher gas
      });
      
      console.log("Current gas limit:", limit);
      setGasLimit(limit);
    } catch (error) {
      console.error("Error fetching gas limit:", error.message);
      
      // Detailed debugging
      if (error.message.includes("Returned values aren't valid")) {
        console.error("This could indicate the contract doesn't have the expected method or the ABI is incorrect");
        
        // List available methods
        console.log("Available contract methods:", Object.keys(contract.methods));
        
        // Try checking if the contract has other expected methods
        try {
          const owner = await contract.methods.owner().call();
          console.log("Contract owner:", owner);
          alert("Contract exists but currentGasLimit method failed. Check console for details.");
        } catch (secondError) {
          console.error("Secondary method also failed:", secondError);
          alert("Contract communication completely failed. You may be on the wrong network.");
        }
      }
    }
  };


  // Update gas limit
  const updateGasLimit = async () => {
    if (contract) {
      const newLimit = BigInt(gasLimit) * BigInt(110) / BigInt(100);  // Increase by 10%
      await contract.methods.setGasLimit(newLimit).send({ from: account });
      alert("Gas limit updated successfully!");
      fetchGasLimit();  // Refresh the displayed gas limit
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