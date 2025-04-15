// src/config.ts
export type NetworkConfig = {
    TYPE: string;
    URL: string;
    GRAPHQL_URL?: string;
  };
  
  export type ContractConfig = {
    PACKAGE_ID: string;
    MODULE_NAME: string;
  };
  
  let currentNetwork: NetworkConfig = {
    TYPE: "testnet",
    URL: "https://fullnode.testnet.sui.io",
  };
  
  let currentContract: ContractConfig = {
    PACKAGE_ID: "",
    MODULE_NAME: "",
  };
  
  let gasBudget = 1_200_000_000;
  
  export const setSuiNetwork = (config: NetworkConfig) => {
    currentNetwork = config;
  };
  
  export const getSuiNetwork = () => currentNetwork;
  
  export const setContractConfig = (config: ContractConfig) => {
    currentContract = config;
  };
  
  export const getContractConfig = () => currentContract;
  
  export const setGasBudget = (budget: number) => {
    gasBudget = budget;
  };
  
  export const getGasBudget = () => gasBudget;
  