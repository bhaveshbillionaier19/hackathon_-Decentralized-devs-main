require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();


module.exports = {
  solidity: "0.8.18", // or your preferred Solidity version
  networks: {
    ganache: {
      url: "HTTP://127.0.0.1:7545", // Ganache local blockchain URL
      accounts: [
        // Use the private keys from Ganache accounts, OR simply use mnemonic from Ganache if needed.
        "0xe5ac2a4b47669e9515d36aa7e1ef227c88a0be46269f864f03cc3ee67ede6de6",
        
        // Add more keys if needed
      ],
       // Adjust gas price if necessary
      chainId: 1337 // Make sure it matches Ganache's chainId
    }
  }
};