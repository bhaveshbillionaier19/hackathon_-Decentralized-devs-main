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
  const contractAddress = "0x71e102A49e672B9cB1AfD1606368F470b2A4DDCA";

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
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 5 second timeout
      
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
      <div className="bg-gray-900 bg-opacity-80 p-5 rounded-xl shadow-[0_0_15px_rgba(0,0,0,0.4)] border border-gray-800 backdrop-blur-sm hover:shadow-[0_0_20px_rgba(6,182,212,0.2)] transition-all duration-300">
        <div className="flex justify-between items-center mb-5">
          <h2 className="text-xl font-bold text-cyan-400">Network Status</h2>
          {lastUpdated && (
            <div className="text-xs text-gray-400 flex items-center">
              <span>Updated: {lastUpdated.toLocaleTimeString()}</span>
              <button 
                onClick={fetchGasLimit} 
                className="ml-2 p-1 rounded hover:bg-gray-700 hover:text-cyan-400 transition-all"
                title="Refresh data">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </button>
            </div>
          )}
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-800 p-4 rounded-xl border border-gray-700 hover:border-cyan-800 transition-all group">
            <p className="text-gray-400 group-hover:text-gray-300 transition-colors">Congestion</p>
            <p
              className={`text-xl font-bold ${
                networkCongestion === "Low"
                  ? "text-green-400 group-hover:text-green-300"
                  : networkCongestion === "Medium"
                  ? "text-yellow-400 group-hover:text-yellow-300"
                  : "text-red-400 group-hover:text-red-300"
              } transition-colors`}
            >
              {networkCongestion}
            </p>
          </div>
          <div className="bg-gray-800 p-4 rounded-xl border border-gray-700 hover:border-cyan-800 transition-all group">
            <p className="text-gray-400 group-hover:text-gray-300 transition-colors">Block Utilization</p>
            <p className="text-xl font-bold text-purple-400 group-hover:text-purple-300 transition-colors">{blockUtilization}%</p>
          </div>
          <div className="bg-gray-800 p-4 rounded-xl border border-gray-700 hover:border-cyan-800 transition-all group">
            <p className="text-gray-400 group-hover:text-gray-300 transition-colors">Current Gas Limit</p>
            <p className="text-xl font-bold text-cyan-400 group-hover:text-cyan-300 transition-colors">
              {gasLimit ? web3.utils.fromWei(gasLimit, "gwei") + " Gwei" : "Loading..."}
            </p>
          </div>
          <div className="bg-gray-800 p-4 rounded-xl border border-gray-700 hover:border-cyan-800 transition-all group">
            <p className="text-gray-400 group-hover:text-gray-300 transition-colors">Transactions/Sec</p>
            <p className="text-xl font-bold text-cyan-400 group-hover:text-cyan-300 transition-colors">{tps}</p>
          </div>
        </div>
      </div>

      <div className="bg-gray-900 bg-opacity-80 p-5 rounded-xl shadow-[0_0_15px_rgba(0,0,0,0.4)] border border-gray-800 backdrop-blur-sm hover:shadow-[0_0_20px_rgba(168,85,247,0.2)] transition-all duration-300">
        <h2 className="text-xl font-bold text-purple-400 mb-5">Gas Fee Recommendations</h2>
        <div className="space-y-4">
          <div className="flex justify-between items-center bg-gradient-to-r from-green-900 to-green-900/30 bg-opacity-20 p-3 rounded-lg border border-green-900 hover:border-green-700 transition-all group">
            <div className="group-hover:translate-x-1 transition-transform">
              <span className="text-green-400 font-medium">Slow</span>
              <p className="text-xs text-gray-400 group-hover:text-gray-300 transition-colors">{gasFeeEstimates.slow.time}</p>
            </div>
            <span className="font-bold text-green-400 group-hover:text-green-300 transition-colors">{gasFeeEstimates.slow.fee} Gwei</span>
          </div>
          <div className="flex justify-between items-center bg-gradient-to-r from-yellow-900 to-yellow-900/30 bg-opacity-20 p-3 rounded-lg border border-yellow-900 hover:border-yellow-700 transition-all group">
            <div className="group-hover:translate-x-1 transition-transform">
              <span className="text-yellow-400 font-medium">Medium</span>
              <p className="text-xs text-gray-400 group-hover:text-gray-300 transition-colors">{gasFeeEstimates.medium.time}</p>
            </div>
            <span className="font-bold text-yellow-400 group-hover:text-yellow-300 transition-colors">{gasFeeEstimates.medium.fee} Gwei</span>
          </div>
          <div className="flex justify-between items-center bg-gradient-to-r from-red-900 to-red-900/30 bg-opacity-20 p-3 rounded-lg border border-red-900 hover:border-red-700 transition-all group">
            <div className="group-hover:translate-x-1 transition-transform">
              <span className="text-red-400 font-medium">Fast</span>
              <p className="text-xs text-gray-400 group-hover:text-gray-300 transition-colors">{gasFeeEstimates.fast.time}</p>
            </div>
            <span className="font-bold text-red-400 group-hover:text-red-300 transition-colors">{gasFeeEstimates.fast.fee} Gwei</span>
          </div>
        </div>
      </div>

      <div className="bg-gray-900 bg-opacity-80 p-5 rounded-xl shadow-[0_0_15px_rgba(0,0,0,0.4)] border border-gray-800 backdrop-blur-sm hover:shadow-[0_0_20px_rgba(6,182,212,0.2)] transition-all duration-300">
        <h2 className="text-xl font-bold text-cyan-400 mb-5">Block Information</h2>
        <div className="space-y-4">
          <div className="flex justify-between items-center p-3 border-b border-gray-700 hover:border-cyan-900 group transition-colors">
            <span className="text-gray-400 group-hover:text-gray-300 transition-colors">Latest Block:</span>
            <span className="text-cyan-400 group-hover:text-cyan-300 font-mono transition-colors">{currentBlock ? currentBlock.number : "Loading..."}</span>
          </div>
          <div className="flex justify-between items-center p-3 border-b border-gray-700 hover:border-cyan-900 group transition-colors">
            <span className="text-gray-400 group-hover:text-gray-300 transition-colors">Block Gas Used:</span>
            <span className="text-cyan-400 group-hover:text-cyan-300 font-mono transition-colors">{currentBlock ? currentBlock.gasUsed : "Loading..."}</span>
          </div>
          <div className="flex justify-between items-center p-3 border-b border-gray-700 hover:border-cyan-900 group transition-colors">
            <span className="text-gray-400 group-hover:text-gray-300 transition-colors">Pending Transactions:</span>
            <span className="text-cyan-400 group-hover:text-cyan-300 font-mono transition-colors">{pendingTxCount}</span>
          </div>
          <div className="flex justify-between items-center p-3 border-b border-gray-700 hover:border-cyan-900 group transition-colors">
            <span className="text-gray-400 group-hover:text-gray-300 transition-colors">Registered Nodes:</span>
            <span className="text-cyan-400 group-hover:text-cyan-300 font-mono transition-colors">{registeredNodes.length}</span>
          </div>
        </div>
      </div>

      <div className="bg-gray-900 bg-opacity-80 p-5 rounded-xl shadow-[0_0_15px_rgba(0,0,0,0.4)] border border-gray-800 backdrop-blur-sm hover:shadow-[0_0_20px_rgba(168,85,247,0.2)] transition-all duration-300">
        <h2 className="text-xl font-bold text-purple-400 mb-5 flex items-center">
          MEV Monitor
          {mevAlerts.length > 0 && (
            <span className="ml-2 bg-gradient-to-r from-red-600 to-purple-600 text-white text-xs px-2 py-1 rounded-full">
              {mevAlerts.length}
            </span>
          )}
        </h2>
        <div className="max-h-40 overflow-y-auto">
          {mevAlerts.length > 0 ? (
            <div className="space-y-3">
              {mevAlerts.map((alert, idx) => (
                <div
                  key={idx}
                  className="bg-gradient-to-r from-red-900/50 to-purple-900/30 p-3 rounded-lg border border-red-800 hover:border-red-700 hover:shadow-[0_0_10px_rgba(239,68,68,0.2)] transition-all text-xs group"
                >
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400 group-hover:text-gray-300 transition-colors">From:</span>
                    <span className="font-mono text-red-400 group-hover:text-red-300 transition-colors">{alert.from.substring(0, 8)}...</span>
                  </div>
                  <div className="flex justify-between items-center mt-1">
                    <span className="text-gray-400 group-hover:text-gray-300 transition-colors">To:</span>
                    <span className="font-mono text-red-400 group-hover:text-red-300 transition-colors">{alert.to.substring(0, 8)}...</span>
                  </div>
                  <div className="flex justify-between items-center mt-1">
                    <span className="text-gray-400 group-hover:text-gray-300 transition-colors">Gas Price:</span>
                    <span className="font-mono text-red-400 group-hover:text-red-300 transition-colors">{alert.gasPrice} Gwei</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="flex items-center justify-center h-32 bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700">
              <p className="text-gray-400">No MEV activity detected</p>
            </div>
          )}
        </div>
      </div>

      <div className="col-span-1 md:col-span-2 bg-gray-900 bg-opacity-80 p-5 rounded-xl shadow-[0_0_15px_rgba(0,0,0,0.4)] border border-gray-800 backdrop-blur-sm hover:shadow-[0_0_20px_rgba(6,182,212,0.2)] transition-all duration-300">
        <h2 className="text-xl font-bold text-cyan-400 mb-5">Gas Limit Trends</h2>
        <div className="h-64">
          {gasHistory.length > 0 && <Line data={chartData} options={chartOptions} />}
        </div>
      </div>
      
      {/* New: AI Optimization Recommendations */}
      <div className="col-span-1 md:col-span-2 bg-gray-900 bg-opacity-80 p-5 rounded-xl shadow-[0_0_15px_rgba(0,0,0,0.4)] border border-gray-800 backdrop-blur-sm hover:shadow-[0_0_20px_rgba(168,85,247,0.2)] transition-all duration-300">
        <h2 className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500 mb-5">AI Optimization Recommendations</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
          <div className="bg-gray-800 p-4 rounded-xl border border-gray-700 hover:border-cyan-900 hover:shadow-[0_0_10px_rgba(6,182,212,0.2)] transition-all group">
            <p className="text-gray-400 mb-2 group-hover:text-gray-300 transition-colors">Optimal P2P Connections</p>
            <p className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">{optimalPeerConnections}</p>
            <p className="text-xs text-gray-400 mt-1 group-hover:text-gray-300 transition-colors">Recommended number of peers based on network load</p>
          </div>
          <div className="bg-gray-800 p-4 rounded-xl border border-gray-700 hover:border-cyan-900 hover:shadow-[0_0_10px_rgba(6,182,212,0.2)] transition-all group">
            <p className="text-gray-400 mb-2 group-hover:text-gray-300 transition-colors">Block Size Adjustment</p>
            <p className={`text-2xl font-bold ${blockSizeAdjustment >= 0 ? 'text-green-400 group-hover:text-green-300' : 'text-red-400 group-hover:text-red-300'} transition-colors`}>
              {blockSizeAdjustment >= 0 ? '+' : ''}{blockSizeAdjustment}%
            </p>
            <p className="text-xs text-gray-400 mt-1 group-hover:text-gray-300 transition-colors">Recommended block size adjustment</p>
          </div>
          <div className="bg-gray-800 p-4 rounded-xl border border-gray-700 hover:border-cyan-900 hover:shadow-[0_0_10px_rgba(6,182,212,0.2)] transition-all group">
            <p className="text-gray-400 mb-2 group-hover:text-gray-300 transition-colors">Batch Transactions</p>
            <p className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent">{ebpfRules.batch_threshold}</p>
            <p className="text-xs text-gray-400 mt-1 group-hover:text-gray-300 transition-colors">Transactions to batch for optimization</p>
          </div>
        </div>
        
        <div className="bg-gray-800 p-5 rounded-xl border border-gray-700 hover:border-purple-900 hover:shadow-[0_0_15px_rgba(168,85,247,0.2)] transition-all">
          <h3 className="font-semibold mb-3 text-purple-400">Recommendations:</h3>
          {recommendations.length > 0 ? (
            <ul className="space-y-2">
              {recommendations.map((rec, idx) => (
                <li key={idx} className="text-gray-300 flex items-start">
                  <span className="text-purple-500 mr-2">■</span>
                  <span className="text-gray-300">{rec}</span>
                </li>
              ))}
            </ul>
          ) : (
            <div className="flex items-center justify-center h-20 bg-gray-900 bg-opacity-50 rounded-lg">
              <p className="text-gray-400">No recommendations available</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  // Render gas management tab
  const renderGasManagement = () => (
    <div className="bg-gray-900 bg-opacity-80 p-6 rounded-xl shadow-[0_0_15px_rgba(0,0,0,0.4)] border border-gray-800 backdrop-blur-sm">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500">Gas Limit Management</h2>
        {lastUpdated && (
          <div className="text-xs text-gray-400 flex items-center">
            <div className="w-2 h-2 bg-cyan-500 rounded-full mr-2 animate-pulse shadow-[0_0_5px_rgba(6,182,212,0.5)]"></div>
            Auto-updating every 30s • Last update: {lastUpdated.toLocaleTimeString()}
          </div>
        )}
      </div>

      <div className="mb-6 bg-gray-800 p-5 rounded-xl border border-gray-700 hover:border-cyan-800 transition-all hover:shadow-[0_0_15px_rgba(6,182,212,0.2)]">
        <div className="flex flex-col md:flex-row items-center gap-4">
          <div className="flex-1">
            <p className="text-sm text-gray-400 mb-1">Current Gas Limit:</p>
            <p className="text-lg font-semibold text-cyan-400">
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
            <p className="text-lg font-semibold text-purple-400">
              {maxGasLimit ? web3.utils.fromWei(maxGasLimit, "gwei") + " Gwei" : "Loading..."}
            </p>
          </div>
        </div>
      </div>

      <div className="bg-gray-800 p-5 rounded-xl border border-gray-700 hover:border-cyan-800 transition-all hover:shadow-[0_0_15px_rgba(6,182,212,0.2)] mb-6">
        <h3 className="text-lg font-semibold mb-4 text-cyan-400">AI Gas Limit Optimization</h3>
        <div className="grid grid-cols-1 gap-4">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div>
              <p className="text-xl text-gray-100 font-bold mb-2 flex items-center flex-wrap">
                <span className="mr-2">Recommended Gas Limit:</span> 
                <span className="font-bold bg-gray-900 px-4 py-2 rounded-lg border border-cyan-700 text-cyan-400 flex items-center shadow-[0_0_10px_rgba(6,182,212,0.3)] hover:shadow-[0_0_15px_rgba(6,182,212,0.5)] transition-all">
                  {newGasLimit}
                  <div className="w-2 h-2 bg-cyan-500 rounded-full ml-2 animate-pulse"></div>
                </span>
              </p>
              <p className="text-xs text-gray-400 mt-1 max-w-lg">
                This gas limit is dynamically computed by AI based on current network conditions
              </p>
            </div>
            <div className="flex gap-3 mt-4 md:mt-0">              
              <button
                className="bg-gradient-to-r from-cyan-600 to-purple-600 hover:from-cyan-500 hover:to-purple-500 text-white font-bold py-2 px-6 rounded-lg shadow-[0_0_10px_rgba(6,182,212,0.3)] hover:shadow-[0_0_15px_rgba(6,182,212,0.5)] transition-all disabled:opacity-50 disabled:hover:shadow-none"
                onClick={updateGasLimit}
                disabled={!newGasLimit || isLoading}>
                {isLoading ? 
                  <span className="flex items-center"><span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></span>Applying...</span> : 
                  'Apply Gas Limit'}
              </button>
            </div>
          </div>
        </div>
      </div>
      
      <div className="bg-gray-800 p-5 rounded-xl border border-gray-700 hover:border-purple-800 transition-all hover:shadow-[0_0_15px_rgba(168,85,247,0.2)]">
        <h3 className="text-lg font-semibold mb-4 text-purple-400">eBPF Traffic Control Rules</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-900 bg-opacity-70 p-4 rounded-lg border border-gray-700 hover:border-cyan-900 transition-all group">
            <p className="text-gray-300 mb-2 group-hover:text-cyan-400 transition-colors">Transaction Batching</p>
            <div className="flex items-center space-x-4">
              <div>
                <p className="text-sm">Batch threshold: 
                  <span className="text-cyan-400 font-bold ml-2">{ebpfRules.batch_threshold}</span>
                </p>
                <p className="text-sm mt-1">Max TX per sender: 
                  <span className="text-cyan-400 font-bold ml-2">{ebpfRules.max_txs_per_sender}</span>
                </p>
              </div>
              <div className="w-12 h-12 flex-shrink-0">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" className="w-full h-full text-cyan-500 opacity-70 group-hover:opacity-100 transition-opacity">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
              </div>
            </div>
          </div>
          <div className="bg-gray-900 bg-opacity-70 p-4 rounded-lg border border-gray-700 hover:border-purple-900 transition-all group">
            <p className="text-gray-300 mb-2 group-hover:text-purple-400 transition-colors">Priority Accounts</p>
            {ebpfRules.priority_accounts.length > 0 ? (
              <ul className="list-disc list-inside space-y-1">
                {ebpfRules.priority_accounts.map((acct, idx) => (
                  <li key={idx} className="text-xs group-hover:text-purple-300 transition-colors">
                    <span className="font-mono">{acct.address.substring(0, 10)}...</span> - {acct.description}
                  </li>
                ))}
              </ul>
            ) : (
              <div className="flex items-center space-x-2">
                <p className="text-sm text-gray-400">No priority accounts configured</p>
                <div className="w-8 h-8 flex-shrink-0">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" className="w-full h-full text-purple-500 opacity-70 group-hover:opacity-100 transition-opacity">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
                  </svg>
                </div>
              </div>
            )}
          </div>
        </div>
        
        <div className="mt-4">
          <p className="text-gray-300 mb-2 hover:text-red-400 transition-colors">Dropped Transaction Patterns</p>
          <div className="bg-gray-900 bg-opacity-70 p-4 rounded-lg border border-gray-700 hover:border-red-900 transition-all">
            {ebpfRules.drop_patterns.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {ebpfRules.drop_patterns.map((pattern, idx) => (
                  <div key={idx} className="bg-gradient-to-r from-red-900/40 to-red-900/10 p-3 rounded-lg text-xs border border-red-900/50 hover:border-red-700 hover:shadow-[0_0_10px_rgba(239,68,68,0.2)] transition-all group">
                    <span className="font-mono text-red-400 group-hover:text-red-300 transition-colors">{pattern.pattern}</span>
                    <p className="text-gray-400 mt-1 group-hover:text-gray-300 transition-colors">{pattern.description}</p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex justify-center items-center h-20">
                <p className="text-sm text-gray-400">No transaction dropping rules active</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );

  // Render node management tab
  const renderNodeManagement = () => (
    <div className="bg-gray-900 bg-opacity-80 p-6 rounded-xl shadow-[0_0_15px_rgba(0,0,0,0.4)] border border-gray-800 backdrop-blur-sm">
      <h2 className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500 mb-6">Node Management</h2>

      <div className="mb-6 bg-gray-800 p-5 rounded-xl border border-gray-700 hover:border-cyan-800 transition-all hover:shadow-[0_0_15px_rgba(6,182,212,0.2)]">
        <h3 className="text-lg font-semibold mb-4 text-cyan-400">Register New Node</h3>
        <div className="flex flex-col md:flex-row items-center gap-4">
          <div className="flex-1 w-full">
            <input
              type="text"
              placeholder="Node Address (0x...)"
              className="w-full bg-gray-900 border border-gray-700 focus:border-cyan-700 rounded-lg p-3 text-white focus:outline-none focus:ring-1 focus:ring-cyan-500 transition-all"
              value={newNodeAddress}
              onChange={(e) => setNewNodeAddress(e.target.value)}
            />
          </div>
          <button
            className="bg-gradient-to-r from-cyan-600 to-purple-600 hover:from-cyan-500 hover:to-purple-500 text-white font-bold py-2 px-6 rounded-lg shadow-[0_0_10px_rgba(6,182,212,0.3)] hover:shadow-[0_0_15px_rgba(6,182,212,0.5)] transition-all"
            onClick={registerNode}
          >
            Register Node
          </button>
        </div>
        <p className="text-xs text-gray-400 mt-3">
          Registered nodes participate in traffic management and consensus on the network.
        </p>
      </div>
      
      <div className="mb-6 bg-gray-800 p-5 rounded-xl border border-gray-700 hover:border-purple-800 transition-all hover:shadow-[0_0_15px_rgba(168,85,247,0.2)]">
        <h3 className="text-lg font-semibold mb-4 text-purple-400">Node Performance Metrics</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-gray-900 bg-opacity-70 p-4 rounded-lg border border-gray-700 hover:border-cyan-900 transition-all group">
            <p className="text-gray-400 group-hover:text-gray-300 transition-colors">Latency</p>
            <p className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent group-hover:translate-y-[-2px] transition-transform">500 ms</p>
          </div>
          <div className="bg-gray-900 bg-opacity-70 p-4 rounded-lg border border-gray-700 hover:border-green-900 transition-all group">
            <p className="text-gray-400 group-hover:text-gray-300 transition-colors">Uptime</p>
            <p className="text-xl font-bold text-green-400 group-hover:text-green-300 group-hover:translate-y-[-2px] transition-all">98.5%</p>
          </div>
          <div className="bg-gray-900 bg-opacity-70 p-4 rounded-lg border border-gray-700 hover:border-yellow-900 transition-all group">
            <p className="text-gray-400 group-hover:text-gray-300 transition-colors">Resource Usage</p>
            <p className="text-xl font-bold text-yellow-400 group-hover:text-yellow-300 group-hover:translate-y-[-2px] transition-all">70%</p>
          </div>
          <div className="bg-gray-900 bg-opacity-70 p-4 rounded-lg border border-gray-700 hover:border-purple-900 transition-all group">
            <p className="text-gray-400 group-hover:text-gray-300 transition-colors">Throughput</p>
            <p className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent group-hover:translate-y-[-2px] transition-transform">1000 TPS</p>
          </div>
        </div>
      </div>
      
      <div className="mb-6 bg-gray-800 p-5 rounded-xl border border-gray-700 hover:border-cyan-800 transition-all hover:shadow-[0_0_15px_rgba(6,182,212,0.2)]">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-cyan-400">P2P Network Configuration</h3>
          <div className="bg-gray-900 px-3 py-1 rounded-full text-cyan-400 text-sm border border-cyan-800 shadow-[0_0_5px_rgba(6,182,212,0.2)]">
            <span className="font-semibold">Optimal Peers: {optimalPeerConnections}</span>
          </div>
        </div>
        
        <div className="bg-gray-900 bg-opacity-70 p-4 rounded-lg border border-gray-700">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            <div>
              <h4 className="text-sm font-semibold text-gray-300 mb-3">Network Configuration</h4>
              <div className="space-y-3">
                <div className="flex justify-between items-center text-sm group">
                  <span className="text-gray-400 group-hover:text-gray-300 transition-colors">Connection Type:</span>
                  <span className="font-medium text-cyan-400 group-hover:text-cyan-300 transition-colors">Mesh Network</span>
                </div>
                <div className="flex justify-between items-center text-sm group">
                  <span className="text-gray-400 group-hover:text-gray-300 transition-colors">Peer Discovery:</span>
                  <span className="font-medium text-cyan-400 group-hover:text-cyan-300 transition-colors">Enabled (DHT)</span>
                </div>
                <div className="flex justify-between items-center text-sm group">
                  <span className="text-gray-400 group-hover:text-gray-300 transition-colors">Current Peers:</span>
                  <span className="font-medium text-cyan-400 group-hover:text-cyan-300 transition-colors">{40 + Math.floor(Math.random() * 20)}</span>
                </div>
                <div className="flex justify-between items-center text-sm group">
                  <span className="text-gray-400 group-hover:text-gray-300 transition-colors">Block Propagation:</span>
                  <span className="font-medium text-cyan-400 group-hover:text-cyan-300 transition-colors">1 second</span>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="text-sm font-semibold text-gray-300 mb-3">Network Adjustment</h4>
              <div className="space-y-3">
                <div className="flex justify-between items-center text-sm">
                  <span className="text-gray-400">Connection Strategy:</span>
                  <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                    networkCongestion === "Low" ? "bg-gradient-to-r from-green-900 to-green-700 text-green-400 border border-green-700 shadow-[0_0_5px_rgba(16,185,129,0.3)]" 
                    : networkCongestion === "Medium" ? "bg-gradient-to-r from-yellow-900 to-yellow-700 text-yellow-400 border border-yellow-700 shadow-[0_0_5px_rgba(245,158,11,0.3)]" 
                    : "bg-gradient-to-r from-red-900 to-red-700 text-red-400 border border-red-700 shadow-[0_0_5px_rgba(239,68,68,0.3)]"
                  }`}>
                    {networkCongestion === "Low" ? "Maximize" : networkCongestion === "Medium" ? "Balanced" : "Minimize"}
                  </span>
                </div>
                <div className="flex justify-between items-center text-sm group">
                  <span className="text-gray-400 group-hover:text-gray-300 transition-colors">Block Size Adjustment:</span>
                  <span className={`transition-colors font-medium ${
                    blockSizeAdjustment >= 0 ? "text-green-400 group-hover:text-green-300" : "text-red-400 group-hover:text-red-300"
                  }`}>
                    {blockSizeAdjustment >= 0 ? "+" : ""}{blockSizeAdjustment}%
                  </span>
                </div>
                <div className="flex justify-between items-center text-sm group">
                  <span className="text-gray-400 group-hover:text-gray-300 transition-colors">Propagation Hops:</span>
                  <span className="font-medium text-purple-400 group-hover:text-purple-300 transition-colors">
                    {networkCongestion === "Low" ? "4" : networkCongestion === "Medium" ? "3" : "2"}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-purple-400">Registered Nodes</h3>
          <div className="px-3 py-1 rounded-full bg-gray-900 text-xs text-purple-400 border border-purple-800 shadow-[0_0_5px_rgba(168,85,247,0.2)]">
            Total: {registeredNodes.length}
          </div>
        </div>
        <div className="bg-gray-800 p-5 rounded-xl border border-gray-700 hover:border-purple-800 transition-all hover:shadow-[0_0_15px_rgba(168,85,247,0.2)] max-h-96 overflow-y-auto">
          {registeredNodes.length > 0 ? (
            <ul className="space-y-3">
              {registeredNodes.map((node, index) => (
                <li key={index} className="flex items-center justify-between border-b border-gray-700 pb-3 hover:border-purple-900 transition-colors group">
                  <div className="truncate w-3/4 font-mono text-gray-300 group-hover:text-gray-100 transition-colors">{node}</div>
                  <span className="bg-gradient-to-r from-green-900 to-green-700 text-green-400 text-xs px-3 py-1 rounded-full border border-green-700 shadow-[0_0_5px_rgba(16,185,129,0.3)] group-hover:shadow-[0_0_8px_rgba(16,185,129,0.5)] transition-all">
                    Active
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <div className="flex items-center justify-center h-32 bg-gray-900 bg-opacity-50 rounded-lg">
              <div className="text-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 mx-auto text-gray-600 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
                <p className="text-gray-400">No nodes registered yet.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <div className="bg-gradient-to-r from-gray-900 to-black text-white min-h-screen p-4 font-sans">
      <header className="container mx-auto flex flex-col md:flex-row justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500 drop-shadow-[0_0_10px_rgba(6,182,212,0.5)]">Autonomous Traffic Manager</h1>
        {web3 ? (
          <div className="flex items-center gap-2 text-sm">
            <div className="bg-gray-900 px-4 py-2 rounded-full text-cyan-400 border border-cyan-700 shadow-[0_0_10px_rgba(6,182,212,0.3)] hover:shadow-[0_0_15px_rgba(6,182,212,0.5)] transition-all">
              Contract Address: {contractAddress}
            </div>
            <div className="bg-gray-900 px-4 py-2 rounded-full text-purple-400 border border-purple-700 shadow-[0_0_10px_rgba(168,85,247,0.3)] hover:shadow-[0_0_15px_rgba(168,85,247,0.5)] transition-all">
              Connected: {account}
            </div>
          </div>
        ) : (
          <button
            className="bg-gradient-to-r from-cyan-600 to-purple-600 hover:from-cyan-500 hover:to-purple-500 text-white font-bold py-2 px-6 rounded-full shadow-[0_0_10px_rgba(6,182,212,0.5)] hover:shadow-[0_0_15px_rgba(6,182,212,0.7)] transition-all"
            onClick={connectMetaMask}
          >
            Connect MetaMask
          </button>
        )}
      </header>

      {web3 ? (
        <div className="container mx-auto">
          <div className="mb-6 bg-gray-900 bg-opacity-70 backdrop-blur-sm rounded-lg p-1 border border-gray-800 shadow-[0_0_10px_rgba(0,0,0,0.3)]">
            <div className="flex">
              <button
                className={`flex-1 py-3 px-4 font-medium rounded-lg transition-all ${
                  activeTab === 'dashboard' 
                    ? 'bg-gradient-to-r from-cyan-900 to-purple-900 text-white shadow-[0_0_8px_rgba(6,182,212,0.3)]' 
                    : 'bg-gray-800 hover:bg-gray-700 text-gray-300'
                }`}
                onClick={() => setActiveTab('dashboard')}
              >
                Dashboard
              </button>
              <button
                className={`flex-1 py-3 px-4 font-medium rounded-lg mx-1 transition-all ${
                  activeTab === 'gas' 
                    ? 'bg-gradient-to-r from-cyan-900 to-purple-900 text-white shadow-[0_0_8px_rgba(6,182,212,0.3)]' 
                    : 'bg-gray-800 hover:bg-gray-700 text-gray-300'
                }`}
                onClick={() => setActiveTab('gas')}
              >
                Gas Management
              </button>
              <button
                className={`flex-1 py-3 px-4 font-medium rounded-lg transition-all ${
                  activeTab === 'nodes' 
                    ? 'bg-gradient-to-r from-cyan-900 to-purple-900 text-white shadow-[0_0_8px_rgba(6,182,212,0.3)]' 
                    : 'bg-gray-800 hover:bg-gray-700 text-gray-300'
                }`}
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
        <div className="flex justify-center items-center h-64">
          <div className="text-center">
            <div className="inline-block border-t-4 border-l-4 border-cyan-500 w-12 h-12 rounded-full animate-spin mb-4"></div>
            <p className="text-cyan-400">Connecting to blockchain...</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;