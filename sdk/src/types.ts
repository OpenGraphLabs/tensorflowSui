// src/types.ts

export interface DatasetMetadata {
    name: string;
    description?: string;
    tags?: string[];
    dataType: string;
    dataSize: number;
    creator?: string;
    license?: string;
  }
  
  export interface Data {
    path: string;
    annotations: string[];
    blobId: string;
    blobHash: string;
    range?: {
      start?: number;
      end?: number;
    };
  }
  
  export interface DatasetCreationResult {
    datasetId: string;
    dataIds: string[];
  }
  