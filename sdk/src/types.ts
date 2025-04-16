// src/types.ts

export type OpenGraphClientConfig = {
  networkUrl: string;
  packageId: string;
  gasBudget: number;
  walrusNetwork?: string;
  walrusPublisherUrl?: string;
  walrusAggregatorUrl?: string;
};

export interface DatasetMetadata {
    name: string;
    description?: string;
    tags?: string[];
    dataType: string;
    dataSize: number;
    creator?: string;
    license?: string;
  }
  
  export enum WalrusStorageStatus {
    ALREADY_CERTIFIED = "Already certified",
    NEWLY_CREATED = "Newly created",
    UNKNOWN = "Unknown",
  }

  export interface WalrusStorageInfo {
    blobId: string;
    endEpoch: number;
    status: WalrusStorageStatus;
    suiRef: string;
    suiRefType: string;
    mediaUrl: string;
    suiScanUrl: string;
    suiRefId: string;
  }