import React, { useState, useEffect } from "react";
import Web3 from "web3";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

function App() {
  const [web3, setWeb3] = useState(null);
  const [account, setAccount] = useState("");
  const [contract, setContract] = useState(null);
  const [gasLimit, setGasLimit] = useState("");
  const [maxGasLimit, setMaxGasLimit] = useState("");
  const [networkCongestion, setNetworkCongestion] = useState("Medium");
  const [registeredNodes, setRegisteredNodes] = useState([]);
  const [newNodeAddress, setNewNodeAddress] = useState("");
  const [activeTab, setActiveTab] = useState("dashboard");
  const [gasHistory, setGasHistory] = useState([]);
  const [newGasLimit, setNewGasLimit] = useState("");
  const [currentBlock, setCurrentBlock] = useState(null);
  const [pendingTxCount, setPendingTxCount] = useState(0);
  const [tps, setTps] = useState(0);
  const [mevAlerts, setMevAlerts] = useState([]);
  const [gasFeeEstimates, setGasFeeEstimates] = useState({
    slow: { fee: "0", time: "5-10 min" },
    medium: { fee: "0", time: "1-3 min" },
    fast: { fee: "0", time: "< 30 sec" },
  });
  const [blockUtilization, setBlockUtilization] = useState("0");

  // Contract ABI (Ensure this matches your deployed contract)
  const contractABI = [
    {
      inputs: [
        {
          internalType: "uint256",
          name: "_initialGasLimit",
          type: "uint256",
        },
        {
          internalType: "uint256",
          name: "_maxGasLimit",
          type: "uint256",
        },
      ],
      stateMutability: "nonpayable",
      type: "constructor",
    },
    {
      anonymous: false,
      inputs: [
        {
          indexed: false,
          internalType: "uint256",
          name: "newLimit",
          type: "uint256",
        },
      ],
      name: "GasLimitUpdated",
      type: "event",
    },
    {
      anonymous: false,
      inputs: [
        {
          indexed: false,
          internalType: "address",
          name: "nodeAddress",
          type: "address",
        },
      ],
      name: "NodeRegistered",
      type: "event",
    },
    {
      inputs: [],
      name: "currentGasLimit",
      outputs: [
        {
          internalType: "uint256",
          name: "",
          type: "uint256",
        },
      ],
      stateMutability: "view",
      type: "function",
    },
    {
      inputs: [],
      name: "getRegisteredNodes",
      outputs: [
        {
          internalType: "address[]",
          name: "",
          type: "address[]",
        },
      ],
      stateMutability: "view",
      type: "function",
    },
    {
      inputs: [],
      name: "maxGasLimit",
      outputs: [
        {
          internalType: "uint256",
          name: "",
          type: "uint256",
        },
      ],
      stateMutability: "view",
      type: "function",
    },
    {
      inputs: [
        {
          internalType: "address",
          name: "",
          type: "address",
        },
      ],
      name: "nodes",
      outputs: [
        {
          internalType: "address",
          name: "nodeAddress",
          type: "address",
        },
        {
          internalType: "bool",
          name: "isActive",
          type: "bool",
        },
      ],
      stateMutability: "view",
      type: "function",
    },
    {
      inputs: [],
      name: "owner",
      outputs: [
        {
          internalType: "address",
          name: "",
          type: "address",
        },
      ],
      stateMutability: "view",
      type: "function",
    },
    {
      inputs: [
        {
          internalType: "address",
          name: "nodeAddress",
          type: "address",
        },
      ],
      name: "registerNode",
      outputs: [],
      stateMutability: "nonpayable",
      type: "function",
    },
    {
      inputs: [
        {
          internalType: "uint256",
          name: "",
          type: "uint256",
        },
      ],
      name: "registeredNodes",
      outputs: [
        {
          internalType: "address",
          name: "",
          type: "address",
        },
      ],
      stateMutability: "view",
      type: "function",
    },
    {
      inputs: [
        {
          internalType: "uint256",
          name: "newLimit",
          type: "uint256",
        },
      ],
      name: "setGasLimit",
      outputs: [],
      stateMutability: "nonpayable",
      type: "function",
    },
  ];
  const contractAddress = "0xe14bac79c6b465b3c201cbd4f86693ae24cac06c";

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

  // Fetch blockchain data
  const fetchBlockchainData = async () => {
    if (web3 && contract) {
      try {
        // Fetch current gas limit
        const limit = await contract.methods.currentGasLimit().call();
        setGasLimit(limit);

        // Fetch max gas limit
        const maxLimit = await contract.methods.maxGasLimit().call();
        setMaxGasLimit(maxLimit);

        // Get registered nodes
        const nodes = await contract.methods.getRegisteredNodes().call();
        setRegisteredNodes(nodes);

        // Get current block
        const blockNumber = await web3.eth.getBlockNumber();
        const block = await web3.eth.getBlock(blockNumber);
        setCurrentBlock(block);

        // Get pending transactions
        const pendingTx = await web3.eth.getTransactionCount("pending");
        setPendingTxCount(pendingTx);

        // NEW: Get mempool data for better transaction monitoring
        try {
          // Note: Requires an Ethereum node with txpool API enabled
          const txpoolContent = await web3.eth.request({
            method: "txpool_content",
          });

          // Process pending transactions for MEV detection
          if (txpoolContent && txpoolContent.pending) {
            const pendingTxs = [];
            const potentialMEV = [];

            // Iterate through pending transactions
            Object.keys(txpoolContent.pending).forEach((address) => {
              Object.keys(txpoolContent.pending[address]).forEach((nonce) => {
                const tx = txpoolContent.pending[address][nonce];
                pendingTxs.push(tx);

                // MEV detection - identify high gas price transactions to contract addresses
                if (parseInt(tx.gas) > parseInt(limit) * 0.8 && tx.to && tx.input.length > 10) {
                  potentialMEV.push({
                    from: tx.from,
                    to: tx.to,
                    gasPrice: web3.utils.fromWei(tx.gasPrice, "gwei"),
                    value: web3.utils.fromWei(tx.value, "ether"),
                  });
                }
              });
            });

            // Update state with MEV potential transactions
            setMevAlerts(potentialMEV);

            // Calculate more accurate TPS based on mempool data
            setTps(Math.ceil(pendingTxs.length / 15)); // Assuming ~15 sec block time
          }
        } catch (mempoolError) {
          console.log("Mempool data not available:", mempoolError);

          // Fallback: Simulate TPS calculation if mempool data isn't available
          setTps(Math.floor(5 + Math.random() * 15));
        }

        // Update gas history (simulated for demo)
        const newDataPoint = Math.floor(limit * (0.9 + Math.random() * 0.2));
        setGasHistory((prev) => [...prev.slice(-11), newDataPoint]);

        // Calculate network congestion level
        const limitPercentage = (parseInt(limit) / parseInt(maxLimit)) * 100;
        if (limitPercentage < 30) setNetworkCongestion("Low");
        else if (limitPercentage < 70) setNetworkCongestion("Medium");
        else setNetworkCongestion("High");

        // NEW: Estimate optimal gas prices based on network conditions
        const gasPrice = await web3.eth.getGasPrice();
        const baseFeePerGas = block.baseFeePerGas ? block.baseFeePerGas : gasPrice;

        setGasFeeEstimates({
          slow: {
            fee: web3.utils.fromWei(
              web3.utils
                .toBN(baseFeePerGas)
                .mul(web3.utils.toBN(11))
                .div(web3.utils.toBN(10))
                .toString(),
              "gwei"
            ),
            time: "5-10 min",
          },
          medium: {
            fee: web3.utils.fromWei(
              web3.utils
                .toBN(baseFeePerGas)
                .mul(web3.utils.toBN(15))
                .div(web3.utils.toBN(10))
                .toString(),
              "gwei"
            ),
            time: "1-3 min",
          },
          fast: {
            fee: web3.utils.fromWei(
              web3.utils
                .toBN(baseFeePerGas)
                .mul(web3.utils.toBN(20))
                .div(web3.utils.toBN(10))
                .toString(),
              "gwei"
            ),
            time: "< 30 sec",
          },
        });

        // NEW: Calculate block utilization percentage
        const blockUtilization = (
          (parseInt(block.gasUsed) / parseInt(block.gasLimit)) *
          100
        ).toFixed(2);
        setBlockUtilization(blockUtilization);
      } catch (error) {
        console.error("Error fetching blockchain data:", error);
      }
    }
  };

  useEffect(() => {
    async function fetchGasLimit() {
      try {
        const response = await fetch("http://127.0.0.1:5000/predict_gas", {
          method: "POST",
          // headers: {
          //   "Content-Type": "application/json",
          // },
          body: JSON.stringify({}), // sending an empty JSON object
        });
        const result = await response.json();
        console.log("Gas limit prediction result:", result);
        // Assuming the response contains a property like "gasLimit"
        setNewGasLimit(result.gasLimit || "7000000");
      } catch (error) {
        console.error("Error fetching gas limit:", error);
      }
    }
    fetchGasLimit();
  }, []);
  // Update gas limit
  const updateGasLimit = async () => {
    if (contract && newGasLimit) {
      try {
        const limit = parseInt(newGasLimit);
        await contract.methods.setGasLimit(limit).send({ from: account });
        alert("Gas limit updated successfully!");
        fetchBlockchainData();
        setNewGasLimit("");
      } catch (error) {
        console.error("Error updating gas limit:", error);
      }
    } else {
      alert("Please enter a valid gas limit");
    }
  };

  // Register a new node
  const registerNode = async () => {
    if (contract && web3.utils.isAddress(newNodeAddress)) {
      try {
        await contract.methods.registerNode(newNodeAddress).send({ from: account });
        alert("Node registered successfully!");
        fetchBlockchainData();
        setNewNodeAddress("");
      } catch (error) {
        console.error("Error registering node:", error);
      }
    } else {
      alert("Please enter a valid Ethereum address");
    }
  };

  // Set up polling for blockchain data
  useEffect(() => {
    if (web3 && contract) {
      fetchBlockchainData();

      // Set up polling every 10 seconds
      const interval = setInterval(() => {
        fetchBlockchainData();
      }, 10000);

      return () => clearInterval(interval);
    }
  }, [web3, contract]);

  // Chart data
  const chartData = {
    labels: Array(12)
      .fill("")
      .map((_, i) => `${i * 5}m ago`)
      .reverse(),
    datasets: [
      {
        label: "Gas Limit History",
        data: gasHistory,
        borderColor: "rgba(75, 192, 192, 1)",
        backgroundColor: "rgba(75, 192, 192, 0.2)",
        tension: 0.4,
      },
    ],
  };

  // Chart options
  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
      },
      title: {
        display: true,
        text: "Gas Limit Trends",
        color: "white",
      },
    },
    scales: {
      y: {
        ticks: { color: "white" },
      },
      x: {
        ticks: { color: "white" },
      },
    },
  };

  // Render dashboard tab
  const renderDashboard = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="bg-gray-800 p-4 rounded-lg shadow">
        <h2 className="text-xl font-bold mb-4">Network Status</h2>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-700 p-3 rounded-lg">
            <p className="text-gray-400">Congestion</p>
            <p
              className={`text-xl font-bold ${
                networkCongestion === "Low"
                  ? "text-green-400"
                  : networkCongestion === "Medium"
                  ? "text-yellow-400"
                  : "text-red-400"
              }`}
            >
              {networkCongestion}
            </p>
          </div>
          <div className="bg-gray-700 p-3 rounded-lg">
            <p className="text-gray-400">Block Utilization</p>
            <p className="text-xl font-bold">{blockUtilization}%</p>
          </div>
          <div className="bg-gray-700 p-3 rounded-lg">
            <p className="text-gray-400">Current Gas Limit</p>
            <p className="text-xl font-bold">
              {gasLimit ? web3.utils.fromWei(gasLimit, "gwei") + " Gwei" : "Loading..."}
            </p>
          </div>
          <div className="bg-gray-700 p-3 rounded-lg">
            <p className="text-gray-400">Transactions/Sec</p>
            <p className="text-xl font-bold">{tps}</p>
          </div>
        </div>
      </div>

      <div className="bg-gray-800 p-4 rounded-lg shadow">
        <h2 className="text-xl font-bold mb-4">Gas Fee Recommendations</h2>
        <div className="space-y-3">
          <div className="flex justify-between items-center bg-green-900 bg-opacity-20 p-2 rounded border border-green-700">
            <div>
              <span className="text-green-400 font-medium">Slow</span>
              <p className="text-xs text-gray-400">{gasFeeEstimates.slow.time}</p>
            </div>
            <span className="font-bold">{gasFeeEstimates.slow.fee} Gwei</span>
          </div>
          <div className="flex justify-between items-center bg-yellow-900 bg-opacity-20 p-2 rounded border border-yellow-700">
            <div>
              <span className="text-yellow-400 font-medium">Medium</span>
              <p className="text-xs text-gray-400">{gasFeeEstimates.medium.time}</p>
            </div>
            <span className="font-bold">{gasFeeEstimates.medium.fee} Gwei</span>
          </div>
          <div className="flex justify-between items-center bg-red-900 bg-opacity-20 p-2 rounded border border-red-700">
            <div>
              <span className="text-red-400 font-medium">Fast</span>
              <p className="text-xs text-gray-400">{gasFeeEstimates.fast.time}</p>
            </div>
            <span className="font-bold">{gasFeeEstimates.fast.fee} Gwei</span>
          </div>
        </div>
      </div>

      <div className="bg-gray-800 p-4 rounded-lg shadow">
        <h2 className="text-xl font-bold mb-4">Block Information</h2>
        <div className="space-y-2">
          <div className="flex justify-between">
            <span className="text-gray-400">Latest Block:</span>
            <span>{currentBlock ? currentBlock.number : "Loading..."}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Block Gas Used:</span>
            <span>{currentBlock ? currentBlock.gasUsed : "Loading..."}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Pending Transactions:</span>
            <span>{pendingTxCount}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Registered Nodes:</span>
            <span>{registeredNodes.length}</span>
          </div>
        </div>
      </div>

      <div className="bg-gray-800 p-4 rounded-lg shadow">
        <h2 className="text-xl font-bold mb-4 flex items-center">
          MEV Monitor
          {mevAlerts.length > 0 && (
            <span className="ml-2 bg-red-600 text-white text-xs px-2 py-1 rounded-full">
              {mevAlerts.length}
            </span>
          )}
        </h2>
        <div className="max-h-40 overflow-y-auto">
          {mevAlerts.length > 0 ? (
            <div className="space-y-2">
              {mevAlerts.map((alert, idx) => (
                <div
                  key={idx}
                  className="bg-red-900 bg-opacity-20 p-2 rounded border border-red-700 text-xs"
                >
                  <div className="flex justify-between">
                    <span>From:</span>
                    <span className="font-mono">{alert.from.substring(0, 8)}...</span>
                  </div>
                  <div className="flex justify-between">
                    <span>To:</span>
                    <span className="font-mono">{alert.to.substring(0, 8)}...</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Gas Price:</span>
                    <span>{alert.gasPrice} Gwei</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-400 text-center">No MEV activity detected</p>
          )}
        </div>
      </div>

      <div className="col-span-1 md:col-span-2 bg-gray-800 p-4 rounded-lg shadow">
        <h2 className="text-xl font-bold mb-4">Gas Limit Trends</h2>
        <div className="h-64">
          {gasHistory.length > 0 && <Line data={chartData} options={chartOptions} />}
        </div>
      </div>
    </div>
  );

  // Render gas management tab
  const renderGasManagement = () => (
    <div className="bg-gray-800 p-6 rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">Gas Limit Management</h2>

      <div className="mb-6 bg-gray-700 p-4 rounded-lg">
        <div className="flex flex-col md:flex-row items-center gap-4">
          <div className="flex-1">
            <p className="text-sm text-gray-400 mb-1">Current Gas Limit:</p>
            <p className="text-lg font-semibold">
              {gasLimit ? web3.utils.fromWei(gasLimit, "gwei") + " Gwei" : "Loading..."}
            </p>
          </div>
          <div className="flex-1">
            <p className="text-sm text-gray-400 mb-1">Network Congestion:</p>
            <p
              className={`text-lg font-semibold ${
                networkCongestion === "Low"
                  ? "text-green-400"
                  : networkCongestion === "Medium"
                  ? "text-yellow-400"
                  : "text-red-400"
              }`}
            >
              {networkCongestion}
            </p>
          </div>
          <div className="flex-1">
            <p className="text-sm text-gray-400 mb-1">Maximum Limit:</p>
            <p className="text-lg font-semibold">
              {maxGasLimit ? web3.utils.fromWei(maxGasLimit, "gwei") + " Gwei" : "Loading..."}
            </p>
          </div>
        </div>
      </div>

      <div className="bg-gray-700 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">Updated Gas Limit</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex flex-col md:flex-row items-center gap-3">
            <div className="flex-1">
              {/* <input
              type="number"
              placeholder="New gas limit (in Gwei)"
              className="w-full bg-gray-700 border bor-gray-600 rounded-md p-2 text-white"
              value={newGasLimit}
              onChange={(e) => setNewGasLimit(e.target.value)}
            /> */}
            </div>
            {/* <button
            className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition-colors"
            onClick={updateGasLimit}
          >
            Update Gas Limit
          </button> */}
          </div>
          {/* <p className="text-xs text-gray-400 mt-2">
          Adjusting the gas limit affects how many transactions can be processed in a block. Higher
          limits may allow more transactions but can impact network stability.
        </p> */}
          <div>
            <p className="text-4xl text-gray-100 font-bold mb-1 flex gap-2">
              Recommended Gas Limit:<div className="font-bold text-green-400">{gasLimit}</div>
            </p>
          </div>
        </div>
      </div>
    </div>
  );

  // Render node management tab
  const renderNodeManagement = () => (
    <div className="bg-gray-800 p-6 rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">Node Management</h2>

      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-2">Register New Node</h3>
        <div className="flex flex-col md:flex-row items-center gap-3">
          <div className="flex-1">
            <input
              type="text"
              placeholder="Node Address (0x...)"
              className="w-full bg-gray-700 border border-gray-600 rounded-md p-2 text-white"
              value={newNodeAddress}
              onChange={(e) => setNewNodeAddress(e.target.value)}
            />
          </div>
          <button
            className="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded transition-colors"
            onClick={registerNode}
          >
            Register Node
          </button>
        </div>
        <p className="text-xs text-gray-400 mt-2">
          Registered nodes participate in traffic management and consensus on the network.
        </p>
      </div>

      <div>
        <h3 className="text-lg font-semibold mb-2">Registered Nodes ({registeredNodes.length})</h3>
        <div className="bg-gray-700 p-4 rounded-lg max-h-96 overflow-y-auto">
          {registeredNodes.length > 0 ? (
            <ul className="space-y-2">
              {registeredNodes.map((node, index) => (
                <li
                  key={index}
                  className="flex items-center justify-between border-b border-gray-600 pb-2"
                >
                  <div className="truncate w-3/4">{node}</div>
                  <span className="bg-green-900 text-green-400 text-xs px-2 py-1 rounded">
                    Active
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-gray-400">No nodes registered yet.</p>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <div className="bg-gray-900 text-white min-h-screen p-4">
      <header className="container mx-auto flex flex-col md:flex-row justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-blue-400">Autonomous Traffic Manager</h1>
        {web3 ? (
          <div className="flex items-center gap-2 text-sm">
            <div className="bg-green-800 px-3 py-1 rounded-full text-white border border-white">
              Contact Address: {account}
            </div>
          </div>
        ) : (
          <button
            className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition-colors"
            onClick={connectMetaMask}
          >
            Connect MetaMask
          </button>
        )}
      </header>

      {web3 ? (
        <div className="container mx-auto">
          <div className="mb-6 bg-gray-800 rounded-lg p-1">
            <div className="flex">
              <button
                className={`flex-1 py-2 px-4 rounded-lg transition-colors ${
                  activeTab === "dashboard"
                    ? "bg-blue-600 text-white"
                    : "text-gray-300 hover:bg-gray-700"
                }`}
                onClick={() => setActiveTab("dashboard")}
              >
                Dashboard
              </button>
              <button
                className={`flex-1 py-2 px-4 rounded-lg transition-colors ${
                  activeTab === "gasManagement"
                    ? "bg-blue-600 text-white"
                    : "text-gray-300 hover:bg-gray-700"
                }`}
                onClick={() => setActiveTab("gasManagement")}
              >
                Gas Management
              </button>
              <button
                className={`flex-1 py-2 px-4 rounded-lg transition-colors ${
                  activeTab === "nodeManagement"
                    ? "bg-blue-600 text-white"
                    : "text-gray-300 hover:bg-gray-700"
                }`}
                onClick={() => setActiveTab("nodeManagement")}
              >
                Node Management
              </button>
            </div>
          </div>

          <div className="mb-6">
            {activeTab === "dashboard" && renderDashboard()}
            {activeTab === "gasManagement" && renderGasManagement()}
            {activeTab === "nodeManagement" && renderNodeManagement()}
          </div>
        </div>
      ) : (
        <div className="container mx-auto text-center">
          <div className="bg-gray-800 p-8 rounded-lg shadow-lg max-w-2xl mx-auto">
            <h2 className="text-xl font-bold mb-4">Welcome to Autonomous Traffic Manager</h2>
            <p className="text-gray-400 mb-6">
              Connect your wallet to monitor and optimize blockchain network traffic in real-time.
            </p>
            <button
              className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg transition-colors"
              onClick={connectMetaMask}
            >
              Connect with MetaMask
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
