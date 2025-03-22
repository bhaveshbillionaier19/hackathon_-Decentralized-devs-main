// SPDX-License-Identifier: MIT
pragma solidity ^0.8.18;

contract NetworkManager {
    address public owner;
    uint256 public currentGasLimit;
    uint256 public maxGasLimit;
    
    struct Node {
        address nodeAddress;
        bool isActive;
    }
    
    mapping(address => Node) public nodes;
    address[] public registeredNodes;

    event GasLimitUpdated(uint256 newLimit);
    event NodeRegistered(address nodeAddress);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorized");
        _;
    }

    constructor(uint256 _initialGasLimit, uint256 _maxGasLimit) {
        owner = msg.sender;
        currentGasLimit = _initialGasLimit;
        maxGasLimit = _maxGasLimit;
    }

    // AI Agent callable function
    function setGasLimit(uint256 newLimit) external onlyOwner {
        require(newLimit <= maxGasLimit, "Exceeds max gas limit");
        currentGasLimit = newLimit;
        emit GasLimitUpdated(newLimit);
    }

    // Node registration with approval
    function registerNode(address nodeAddress) external {
        require(!nodes[nodeAddress].isActive, "Node already registered");
        
        nodes[nodeAddress] = Node({
            nodeAddress: nodeAddress,
            isActive: true
        });
        
        registeredNodes.push(nodeAddress);
        emit NodeRegistered(nodeAddress);
    }

    function getRegisteredNodes() external view returns (address[] memory) {
        return registeredNodes;
    }
}