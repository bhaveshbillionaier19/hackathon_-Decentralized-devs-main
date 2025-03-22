// ignition/modules/deploy.js
const { buildModule } = require("@nomicfoundation/hardhat-ignition/modules");

module.exports = buildModule("NetworkManagerModule", (m) => {
  const initialGasLimit = m.getParameter("initialGasLimit", 30_000_000); // 30 million gas
  const maxGasLimit = m.getParameter("maxGasLimit", 50_000_000); // 50 million gas

  const networkManager = m.contract("NetworkManager", [initialGasLimit, maxGasLimit]);

  return { networkManager };
});
// 0x0Eab42a7c6262B0be833Ed4C2dC0070faEa480EF