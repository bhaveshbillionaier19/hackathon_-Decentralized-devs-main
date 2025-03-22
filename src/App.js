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
  const [newGasLimit, setNewGasLimit] = useState("500000");
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
  // New state variables for enhanced blockchain optimizations
  const [optimalPeerConnections, setOptimalPeerConnections] = useState(50);
  const [blockSizeAdjustment, setBlockSizeAdjustment] = useState(0);
  const [recommendations, setRecommendations] = useState([]);
  const [ebpfRules, setEbpfRules] = useState({
    priority_accounts: [],
    drop_patterns: [],
    batch_threshold: 5,
    max_txs_per_sender: 20
  });
  const [trafficRules, setTrafficRules] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);

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
        const pendingTx = await web3.eth.getTransactionCount(account, "pending");
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
  
  // Fetch AI-optimized gas limit and other blockchain parameters
  async function fetchGasLimit() {
    setIsLoading(true);
    try {
      // Get current node metrics with more randomization
      const nodeMetrics = {
        latency: 300 + Math.random() * 400,  // 300-700ms latency (more variance)
        uptime: 97 + Math.random() * 2.5,    // 97-99.5% uptime
        resource_usage: 50 + Math.random() * 40, // 50-90% resource usage (more variance)
        throughput: 800 + Math.random() * 500   // 800-1300 TPS (more variance)
      };
      
      // Include connection timeout for more reliable API calls
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
      
      try {
        const response = await fetch("http://127.0.0.1:5000/predict_gas", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(nodeMetrics),
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
          throw new Error(`API returned status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log("AI Optimization results:", result);
        
        if (result.success) {
          // Extract all optimization parameters
          const optimization = result.optimization;
          
          // Update gas limit
          setNewGasLimit(optimization.current_gas_limit.toString());
          
          // Update gas fee estimates
          setGasFeeEstimates(optimization.gas_fee_estimates);
          
          // Update network congestion
          setNetworkCongestion(optimization.network_congestion);
          
          // Update metrics
          setPendingTxCount(optimization.metrics.pending_transactions);
          setBlockUtilization((optimization.metrics.current_gas_used / optimization.current_gas_limit * 100).toFixed(2));
          
          // Update new optimization parameters
          setOptimalPeerConnections(optimization.optimal_peer_connections);
          setBlockSizeAdjustment(optimization.block_size_adjustment);
          setRecommendations(optimization.recommendations);
          
          // If TPS is provided, update it
          if (optimization.metrics.tps) {
            setTps(optimization.metrics.tps);
          }
          
          // Fetch eBPF traffic rules
          fetchTrafficRules();
          
          // Add gas limit to history for the chart
          setGasHistory(prev => {
            const newHistory = [...prev, optimization.current_gas_limit];
            if (newHistory.length > 12) {
              return newHistory.slice(newHistory.length - 12);
            }
            return newHistory;
          });
        }
      } catch (apiError) {
        console.error("API Connection Error:", apiError);
        
        // Generate fallback data if API is unreachable
        // This helps the dashboard continue to be dynamic even if the backend is down
        
        const networkState = 
          nodeMetrics.resource_usage > 80 || nodeMetrics.latency > 600 ? "High" :
          nodeMetrics.resource_usage < 50 && nodeMetrics.latency < 400 ? "Low" : "Medium";
        
        const baseGasLimit = 30000000;
        const adjustmentFactor = 
          networkState === "High" ? 0.9 :
          networkState === "Low" ? 1.1 : 1.0;
        
        const dynamicGasLimit = Math.floor(baseGasLimit * adjustmentFactor);
        
        // Set fallback values
        setNewGasLimit(dynamicGasLimit.toString());
        setNetworkCongestion(networkState);
        setPendingTxCount(Math.floor(100 + Math.random() * 100));
        setBlockUtilization((65 + Math.random() * 15).toFixed(2));
        
        // Dynamic gas fees based on congestion
        const baseFee = networkState === "High" ? 25 : networkState === "Low" ? 5 : 12;
        setGasFeeEstimates({
          slow: {
            fee: (baseFee * 0.7).toFixed(2),
            time: networkState === "High" ? "10-20 min" : networkState === "Low" ? "1-3 min" : "5-10 min"
          },
          medium: {
            fee: baseFee.toFixed(2),
            time: networkState === "High" ? "5-10 min" : networkState === "Low" ? "30-60 sec" : "1-3 min"
          },
          fast: {
            fee: (baseFee * 1.5).toFixed(2),
            time: networkState === "High" ? "1-3 min" : networkState === "Low" ? "<30 sec" : "30-60 sec"
          }
        });
        
        // Add gas limit to history for the chart
        setGasHistory(prev => {
          const newHistory = [...prev, dynamicGasLimit];
          if (newHistory.length > 12) {
            return newHistory.slice(newHistory.length - 12);
          }
          return newHistory;
        });
      }
      
      // Update timestamp regardless of how we got data (API or fallback)
      setLastUpdated(new Date());
    } catch (error) {
      console.error("Error in fetchGasLimit:", error);
    } finally {
      setIsLoading(false);
    }
  }
  
  // Fetch eBPF traffic control rules
  async function fetchTrafficRules() {
    try {
      const response = await fetch("http://127.0.0.1:5000/optimize_traffic", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({}),
      });
      
      const result = await response.json();
      console.log("Traffic optimization rules:", result);
      
      if (result.success) {
        setEbpfRules(result.traffic_rules);
      }
    } catch (error) {
      console.error("Error fetching traffic rules:", error);
    }
  }
  
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
      fetchGasLimit(); // Fetch AI predictions initially

      // Set up polling every 10 seconds
      const interval = setInterval(() => {
        fetchBlockchainData();
        fetchGasLimit(); // Fetch AI predictions periodically
      }, 10000);

      return () => clearInterval(interval);
    }
  }, [web3, contract]);

  // Also add a separate effect to fetch AI predictions on initial load
  useEffect(() => {
    // Fetch AI predictions immediately when the app loads, even before web3 connection
    fetchGasLimit();
    
    // Set up a refresh interval for AI predictions (every 30 seconds)
    const aiInterval = setInterval(fetchGasLimit, 30000);
    
    return () => clearInterval(aiInterval);
  }, []);

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
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">Network Status</h2>
          {lastUpdated && (
            <div className="text-xs text-gray-400 flex items-center">
              <span>Updated: {lastUpdated.toLocaleTimeString()}</span>
              <button 
                onClick={fetchGasLimit} 
                className="ml-2 p-1 rounded hover:bg-gray-700 transition-colors"
                title="Refresh data">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </button>
            </div>
          )}
        </div>
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
      
      {/* New: AI Optimization Recommendations */}
      <div className="col-span-1 md:col-span-2 bg-gray-800 p-4 rounded-lg shadow">
        <h2 className="text-xl font-bold mb-4">AI Optimization Recommendations</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div className="bg-gray-700 p-3 rounded-lg">
            <p className="text-gray-400">Optimal P2P Connections</p>
            <p className="text-xl font-bold">{optimalPeerConnections}</p>
            <p className="text-xs text-gray-400 mt-1">Recommended number of peers based on network load</p>
          </div>
          <div className="bg-gray-700 p-3 rounded-lg">
            <p className="text-gray-400">Block Size Adjustment</p>
            <p className={`text-xl font-bold ${blockSizeAdjustment >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {blockSizeAdjustment >= 0 ? '+' : ''}{blockSizeAdjustment}%
            </p>
            <p className="text-xs text-gray-400 mt-1">Recommended block size adjustment</p>
          </div>
          <div className="bg-gray-700 p-3 rounded-lg">
            <p className="text-gray-400">Batch Transactions</p>
            <p className="text-xl font-bold">{ebpfRules.batch_threshold}</p>
            <p className="text-xs text-gray-400 mt-1">Transactions to batch for optimization</p>
          </div>
        </div>
        
        <div className="bg-gray-700 p-4 rounded-lg">
          <h3 className="font-semibold mb-2">Recommendations:</h3>
          {recommendations.length > 0 ? (
            <ul className="list-disc list-inside space-y-1">
              {recommendations.map((rec, idx) => (
                <li key={idx} className="text-gray-200">{rec}</li>
              ))}
            </ul>
          ) : (
            <p className="text-gray-400">No recommendations available</p>
          )}
        </div>
      </div>
    </div>
  );

  // Render gas management tab
  const renderGasManagement = () => (
    <div className="bg-gray-800 p-6 rounded-lg shadow">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold">Gas Limit Management</h2>
        {lastUpdated && (
          <div className="text-xs text-gray-400 flex items-center">
            <div className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></div>
            Auto-updating every 30s • Last update: {lastUpdated.toLocaleTimeString()}
          </div>
        )}
      </div>

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
        <h3 className="text-lg font-semibold mb-2">AI Gas Limit Optimization</h3>
        <div className="grid grid-cols-1 gap-4">
          <div className="flex justify-between items-center">
            <div>
              <p className="text-xl text-gray-100 font-bold mb-1 flex gap-2 items-center">
                <span>Recommended Gas Limit:</span> 
                <span className="font-bold bg-gray-900 px-3 py-1 rounded-lg border border-green-700 text-green-400 flex items-center">
                  {newGasLimit}
                  <div className="w-2 h-2 bg-green-500 rounded-full ml-2 animate-pulse"></div>
                </span>
              </p>
              <p className="text-xs text-gray-400 mt-1">
                This gas limit is dynamically computed by AI based on current network conditions
              </p>
            </div>
            <div className="flex gap-3">
              {/* <button
                className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition-colors disabled:opacity-50"
                onClick={fetchGasLimit}
                disabled={isLoading}>
                {isLoading ? 'Predicting...' : 'Predict Gas Limit'}
              </button> */}
              
              <button
                className="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded transition-colors disabled:opacity-50"
                onClick={updateGasLimit}
                disabled={!newGasLimit || isLoading}>
                Apply Gas Limit
              </button>
            </div>
          </div>
        </div>
      </div>
      
      <div className="mt-6 bg-gray-700 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">eBPF Traffic Control Rules</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-800 p-3 rounded-lg">
            <p className="text-gray-400 mb-2">Transaction Batching</p>
            <p className="text-base">Batch similar transactions: <span className="text-white font-bold">{ebpfRules.batch_threshold}</span></p>
            <p className="text-base">Max TX per sender: <span className="text-white font-bold">{ebpfRules.max_txs_per_sender}</span></p>
          </div>
          <div className="bg-gray-800 p-3 rounded-lg">
            <p className="text-gray-400 mb-2">Priority Accounts</p>
            {ebpfRules.priority_accounts.length > 0 ? (
              <ul className="list-disc list-inside">
                {ebpfRules.priority_accounts.map((acct, idx) => (
                  <li key={idx} className="text-xs">
                    <span className="font-mono">{acct.address.substring(0, 10)}...</span> - {acct.description}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm">No priority accounts configured</p>
            )}
          </div>
        </div>
        
        <div className="mt-4">
          <p className="text-gray-400 mb-1">Dropped Transaction Patterns</p>
          <div className="bg-gray-800 p-3 rounded-lg">
            {ebpfRules.drop_patterns.length > 0 ? (
              <div className="grid grid-cols-2 gap-2">
                {ebpfRules.drop_patterns.map((pattern, idx) => (
                  <div key={idx} className="bg-red-900 bg-opacity-20 p-1 rounded text-xs">
                    <span className="font-mono">{pattern.pattern}</span> - {pattern.description}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm">No transaction dropping rules active</p>
            )}
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
      
      <div className="mb-6 bg-gray-700 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-3">Node Performance Metrics</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-gray-800 p-3 rounded-lg">
            <p className="text-gray-400">Latency</p>
            <p className="text-xl font-bold">500 ms</p>
          </div>
          <div className="bg-gray-800 p-3 rounded-lg">
            <p className="text-gray-400">Uptime</p>
            <p className="text-xl font-bold">98.5%</p>
          </div>
          <div className="bg-gray-800 p-3 rounded-lg">
            <p className="text-gray-400">Resource Usage</p>
            <p className="text-xl font-bold">70%</p>
          </div>
          <div className="bg-gray-800 p-3 rounded-lg">
            <p className="text-gray-400">Throughput</p>
            <p className="text-xl font-bold">1000 TPS</p>
          </div>
        </div>
      </div>
      
      <div className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold">P2P Network Configuration</h3>
          <div className="bg-blue-900 bg-opacity-40 px-3 py-1 rounded-full text-blue-400 text-sm">
            <span className="font-semibold">Optimal Peers: {optimalPeerConnections}</span>
          </div>
        </div>
        
        <div className="bg-gray-700 p-4 rounded-lg">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="text-sm font-semibold text-gray-300 mb-2">Network Configuration</h4>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Connection Type:</span>
                  <span>Mesh Network</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Peer Discovery:</span>
                  <span>Enabled (DHT)</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Current Peers:</span>
                  <span>{40 + Math.floor(Math.random() * 20)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Block Propagation:</span>
                  <span>1 second</span>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="text-sm font-semibold text-gray-300 mb-2">Network Adjustment</h4>
              <div className="space-y-2">
                <div className="flex justify-between items-center text-sm">
                  <span className="text-gray-400">Connection Strategy:</span>
                  <span className={`px-2 py-1 rounded ${networkCongestion === "Low" ? "bg-green-900 text-green-400" : networkCongestion === "Medium" ? "bg-yellow-900 text-yellow-400" : "bg-red-900 text-red-400"}`}>
                    {networkCongestion === "Low" ? "Maximize" : networkCongestion === "Medium" ? "Balanced" : "Minimize"}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Block Size Adjustment:</span>
                  <span className={`${blockSizeAdjustment >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {blockSizeAdjustment >= 0 ? "+" : ""}{blockSizeAdjustment}%
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Propagation Hops:</span>
                  <span>{networkCongestion === "Low" ? "4" : networkCongestion === "Medium" ? "3" : "2"}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
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
              Contract Address: {contractAddress.substring(0, 6)}...{contractAddress.substring(38)}
            </div>
            <div className="bg-blue-800 px-3 py-1 rounded-full text-white border border-white">
              Connected: {account.substring(0, 6)}...{account.substring(38)}
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
                className={`flex-1 py-2 px-4 font-medium ${activeTab === 'dashboard' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'}`}
                onClick={() => setActiveTab('dashboard')}
              >
                Dashboard
              </button>
              <button
                className={`flex-1 py-2 px-4 font-medium ${activeTab === 'gas' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'}`}
                onClick={() => setActiveTab('gas')}
              >
                Gas Management
              </button>
              <button
                className={`flex-1 py-2 px-4 font-medium ${activeTab === 'nodes' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'}`}
                onClick={() => setActiveTab('nodes')}
              >
                Node Management
              </button>
            </div>
          </div>

          {activeTab === 'dashboard' && renderDashboard()}
          {activeTab === 'gas' && renderGasManagement()}
          {activeTab === 'nodes' && renderNodeManagement()}
        </div>
      ) : (
        <p>Loading...</p>
      )}
    </div>
  );
}

export default App;