// src/types.ts

export type DatasetSDKConfig = {
  networkUrl: string;
  packageId: string;
  gasBudget: number;
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
  