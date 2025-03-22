// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title Autonomous Traffic Manager
 * @dev Smart contract for managing blockchain traffic and gas limits
 */
contract TrafficManager {
    // Contract owner
    address public owner;
    
    // Gas limit configuration
    uint256 public currentGasLimit;
    uint256 public maxGasLimit;
    
    // Node management
    struct Node {
        address nodeAddress;
        bool isActive;
        uint256 lastHeartbeat;
        uint256 computingPower; // 0-100 scale
        uint256 bandwidth;      // in Mbps
        uint256 latency;        // in milliseconds
    }
    
    // Transaction prioritization
    struct TransactionPriority {
        address contractAddress;
        bool isPriority;
        uint256 priorityLevel; // 1-10 scale
    }
    
    // Block space allocation
    struct BlockSpaceAllocation {
        string txType;         // e.g., "swap", "transfer", "mint"
        uint256 percentage;    // % of block space allocated (0-100)
    }
    
    // Storage variables
    mapping(address => Node) public nodes;
    address[] public registeredNodes;
    mapping(address => TransactionPriority) public priorityContracts;
    address[] public priorityContractsList;
    BlockSpaceAllocation[] public blockSpaceAllocations;
    
    // Network parameters
    uint256 public blockSizeAdjustment;  // +/- percentage adjustment to default block size
    uint256 public mempoolMaxSize;       // Maximum mempool size in transactions
    uint256 public optimalPeerCount;     // Optimal number of peer connections
    uint256 public minGasPrice;          // Minimum gas price to accept transactions
    uint256 public maxTxPerAccount;      // Maximum transactions per account in pending pool
    
    // eBPF Configuration
    mapping(string => bool) public ebpfPatternFilters;
    string[] public ebpfFilterPatterns;
    
    // Events
    event GasLimitUpdated(uint256 newLimit);
    event NodeRegistered(address nodeAddress);
    event NodeDeactivated(address nodeAddress);
    event NodeHeartbeat(address nodeAddress, uint256 timestamp);
    event BlockSpaceAdjusted(string txType, uint256 percentage);
    event EbpfPatternAdded(string pattern);
    event EbpfPatternRemoved(string pattern);
    event PriorityContractAdded(address contractAddress, uint256 priorityLevel);
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier onlyRegisteredNode() {
        require(nodes[msg.sender].isActive, "Only registered active nodes can call this function");
        _;
    }
    
    /**
     * @dev Initialize contract with default gas limit settings
     * @param _initialGasLimit Initial gas limit
     * @param _maxGasLimit Maximum allowed gas limit
     */
    constructor(uint256 _initialGasLimit, uint256 _maxGasLimit) {
        require(_initialGasLimit <= _maxGasLimit, "Initial limit cannot exceed max limit");
        owner = msg.sender;
        currentGasLimit = _initialGasLimit;
        maxGasLimit = _maxGasLimit;
        
        // Default network parameters
        blockSizeAdjustment = 0;         // No adjustment by default
        mempoolMaxSize = 10000;          // Default 10,000 transactions
        optimalPeerCount = 50;           // Default 50 peers
        minGasPrice = 1 gwei;            // Default 1 gwei
        maxTxPerAccount = 20;            // Default 20 transactions per account
        
        // Register the owner as the first node
        registerNode(owner);
    }
    
    /**
     * @dev Set new gas limit (within allowed range)
     * @param newLimit New gas limit value
     */
    function setGasLimit(uint256 newLimit) external onlyOwner {
        require(newLimit <= maxGasLimit, "Gas limit cannot exceed max limit");
        currentGasLimit = newLimit;
        emit GasLimitUpdated(newLimit);
    }
    
    /**
     * @dev Update maximum allowed gas limit
     * @param newMaxLimit New maximum gas limit
     */
    function setMaxGasLimit(uint256 newMaxLimit) external onlyOwner {
        require(newMaxLimit >= currentGasLimit, "Max limit cannot be less than current limit");
        maxGasLimit = newMaxLimit;
    }
    
    /**
     * @dev Register a new node
     * @param nodeAddress Address of the node to register
     */
    function registerNode(address nodeAddress) public {
        if (msg.sender != owner) {
            require(msg.sender == nodeAddress, "Can only register yourself unless you're the owner");
        }
        
        require(!nodes[nodeAddress].isActive, "Node already registered and active");
        
        // Create node record
        Node memory newNode = Node({
            nodeAddress: nodeAddress,
            isActive: true,
            lastHeartbeat: block.timestamp,
            computingPower: 100, // Default value, can be updated later
            bandwidth: 1000,     // Default 1Gbps
            latency: 100         // Default 100ms
        });
        
        // Add node to storage
        nodes[nodeAddress] = newNode;
        
        // Add to list if not already present
        bool isNew = true;
        for (uint i = 0; i < registeredNodes.length; i++) {
            if (registeredNodes[i] == nodeAddress) {
                isNew = false;
                break;
            }
        }
        
        if (isNew) {
            registeredNodes.push(nodeAddress);
        }
        
        emit NodeRegistered(nodeAddress);
    }
    
    /**
     * @dev Deactivate a node
     * @param nodeAddress Address of the node to deactivate
     */
    function deactivateNode(address nodeAddress) external onlyOwner {
        require(nodes[nodeAddress].isActive, "Node is not active");
        nodes[nodeAddress].isActive = false;
        emit NodeDeactivated(nodeAddress);
    }
    
    /**
     * @dev Update node metrics
     * @param computingPower Computing power (0-100)
     * @param bandwidth Bandwidth in Mbps
     * @param latency Latency in milliseconds
     */
    function updateNodeMetrics(uint256 computingPower, uint256 bandwidth, uint256 latency) external onlyRegisteredNode {
        nodes[msg.sender].computingPower = computingPower;
        nodes[msg.sender].bandwidth = bandwidth;
        nodes[msg.sender].latency = latency;
        nodes[msg.sender].lastHeartbeat = block.timestamp;
        
        emit NodeHeartbeat(msg.sender, block.timestamp);
    }
    
    /**
     * @dev Send heartbeat to keep node active
     */
    function sendHeartbeat() external onlyRegisteredNode {
        nodes[msg.sender].lastHeartbeat = block.timestamp;
        emit NodeHeartbeat(msg.sender, block.timestamp);
    }
    
    /**
     * @dev Get all registered nodes
     * @return Array of registered node addresses
     */
    function getRegisteredNodes() external view returns (address[] memory) {
        return registeredNodes;
    }
    
    /**
     * @dev Add a contract to priority list
     * @param contractAddress Address of the contract to prioritize
     * @param priorityLevel Priority level (1-10)
     */
    function addPriorityContract(address contractAddress, uint256 priorityLevel) external onlyOwner {
        require(priorityLevel >= 1 && priorityLevel <= 10, "Priority level must be between 1-10");
        
        // Create or update priority
        bool isNew = !priorityContracts[contractAddress].isPriority;
        priorityContracts[contractAddress] = TransactionPriority({
            contractAddress: contractAddress,
            isPriority: true,
            priorityLevel: priorityLevel
        });
        
        // Add to list if new
        if (isNew) {
            priorityContractsList.push(contractAddress);
        }
        
        emit PriorityContractAdded(contractAddress, priorityLevel);
    }
    
    /**
     * @dev Set block space allocation for transaction types
     * @param txType Transaction type
     * @param percentage Percentage allocation
     */
    function setBlockSpaceAllocation(string calldata txType, uint256 percentage) external onlyOwner {
        require(percentage <= 100, "Percentage cannot exceed 100");
        
        // Check if type already exists
        bool found = false;
        for (uint i = 0; i < blockSpaceAllocations.length; i++) {
            if (keccak256(bytes(blockSpaceAllocations[i].txType)) == keccak256(bytes(txType))) {
                blockSpaceAllocations[i].percentage = percentage;
                found = true;
                break;
            }
        }
        
        // Add new allocation if not found
        if (!found) {
            blockSpaceAllocations.push(BlockSpaceAllocation({
                txType: txType,
                percentage: percentage
            }));
        }
        
        emit BlockSpaceAdjusted(txType, percentage);
    }
    
    /**
     * @dev Set network parameters
     * @param _blockSizeAdjustment Block size adjustment (+/- percentage)
     * @param _mempoolMaxSize Maximum mempool size
     * @param _optimalPeerCount Optimal peer connection count
     * @param _minGasPrice Minimum gas price (in wei)
     * @param _maxTxPerAccount Maximum transactions per account
     */
    function setNetworkParameters(
        int256 _blockSizeAdjustment,
        uint256 _mempoolMaxSize,
        uint256 _optimalPeerCount,
        uint256 _minGasPrice,
        uint256 _maxTxPerAccount
    ) external onlyOwner {
        // Block size adjustment can be negative (decrease) or positive (increase)
        require(_blockSizeAdjustment >= -10 && _blockSizeAdjustment <= 10, "Block size adjustment must be between -10% and +10%");
        
        // If negative, convert to stored format (we'll interpret it as negative in the frontend)
        if (_blockSizeAdjustment < 0) {
            blockSizeAdjustment = uint256(-_blockSizeAdjustment);
        } else {
            blockSizeAdjustment = uint256(_blockSizeAdjustment);
        }
        
        mempoolMaxSize = _mempoolMaxSize;
        optimalPeerCount = _optimalPeerCount;
        minGasPrice = _minGasPrice;
        maxTxPerAccount = _maxTxPerAccount;
    }
    
    /**
     * @dev Add a pattern to eBPF filter
     * @param pattern Pattern to filter
     */
    function addEbpfPattern(string calldata pattern) external onlyOwner {
        require(!ebpfPatternFilters[pattern], "Pattern already exists");
        
        ebpfPatternFilters[pattern] = true;
        ebpfFilterPatterns.push(pattern);
        
        emit EbpfPatternAdded(pattern);
    }
    
    /**
     * @dev Remove a pattern from eBPF filter
     * @param pattern Pattern to remove
     */
    function removeEbpfPattern(string calldata pattern) external onlyOwner {
        require(ebpfPatternFilters[pattern], "Pattern does not exist");
        
        ebpfPatternFilters[pattern] = false;
        
        // Remove from array
        for (uint i = 0; i < ebpfFilterPatterns.length; i++) {
            if (keccak256(bytes(ebpfFilterPatterns[i])) == keccak256(bytes(pattern))) {
                // Move the last item to this position and pop the last item
                ebpfFilterPatterns[i] = ebpfFilterPatterns[ebpfFilterPatterns.length - 1];
                ebpfFilterPatterns.pop();
                break;
            }
        }
        
        emit EbpfPatternRemoved(pattern);
    }
    
    /**
     * @dev Get all active eBPF patterns
     * @return Array of active eBPF patterns
     */
    function getEbpfPatterns() external view returns (string[] memory) {
        return ebpfFilterPatterns;
    }
    
    /**
     * @dev Get all priority contracts
     * @return Array of priority contract addresses
     */
    function getPriorityContracts() external view returns (address[] memory) {
        return priorityContractsList;
    }
    
    /**
     * @dev Get all block space allocations
     * @return Arrays of transaction types and their percentages
     */
    function getBlockSpaceAllocations() external view returns (string[] memory, uint256[] memory) {
        string[] memory types = new string[](blockSpaceAllocations.length);
        uint256[] memory percentages = new uint256[](blockSpaceAllocations.length);
        
        for (uint i = 0; i < blockSpaceAllocations.length; i++) {
            types[i] = blockSpaceAllocations[i].txType;
            percentages[i] = blockSpaceAllocations[i].percentage;
        }
        
        return (types, percentages);
    }
} 