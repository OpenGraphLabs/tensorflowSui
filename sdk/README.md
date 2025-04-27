# OpenGraphClient SDK

> **Easily upload datasets and manage Sui blockchain assets via Walrus Publisher and Aggregator.**

## 📦 Installation

```bash
npm install tensorflow-sui
```
or 
```bash
yarn add tensorflow-sui
```

## 🚀 Quick Start
```typescript
import { OpenGraphClient } from "tensorflow-sui";

// Initialize the client
const client = new OpenGraphClient({
  networkUrl: "https://fullnode.testnet.sui.io",
  packageId: "0x...",
  gasBudget: 100000000,
});

// Upload dataset
const result = await client.uploadDataset(
  files,           // File[]
  walletAddress,   // string
  {
    name: "Example Dataset",
    description: "Sample description",
    dataType: "image",
    dataSize: 12345,
    creator: "John Doe",
    license: "MIT",
    tags: ["sample", "example"],
  },
  ["cat", "dog"],   // Annotations
  3                // Optional epochs
);

// Check storage info and transaction
console.log(result.storageInfos);
console.log(result.transaction);
```

## 📚 API Overview
### Constructor
```typescript
new OpenGraphClient(config: OpenGraphClientConfig);
```
### Constructor Parameters

| Parameter            | Type     | Description                          |
|-----------------------|----------|--------------------------------------|
| `networkUrl`          | `string` | Sui RPC endpoint URL                 |
| `packageId`           | `string` | Smart contract package ID            |
| `gasBudget`           | `number` | Default gas budget                   |
| `walrusNetwork`?      | `string` | Walrus network (default: `testnet`)   |
| `walrusPublisherUrl`? | `string` | Walrus publisher API URL             |
| `walrusAggregatorUrl`? | `string` | Walrus aggregator API URL            |


## Main Methods

### uploadDataset

```typescript
uploadDataset(
  files: File[],
  address: string,
  metadata: DatasetMetadata,
  annotations: string[],
  epochs?: number
): Promise<{
  storageInfos: WalrusStorageInfo[],
  transaction: Transaction
}>
```
* files: List of files to upload (File[])
* address: Owner's wallet address (string)
* metadata: Metadata about the dataset (DatasetMetadata)
* annotations: Annotations for each file (string[])
* epochs?: Optional certification duration (number)

### getDatasets

```typescript
getDatasets(ownerAddress: string): Promise<DatasetMetadata[]>
```
* ownerAddress: The Sui address of the dataset owner (string)

### getDatasetById

```typescript
getDatasetById(datasetId: string): Promise<DatasetMetadata>
```
* datasetId: The object ID of the dataset (string)

### uploadTrainingData

```typescript
uploadTrainingData(file: File, address: string): Promise<WalrusStorageInfo>
```
* file: The training data file to upload (File)
* address: Wallet address for ownership (string)

### getTrainingData

```typescript
getTrainingData(blobIds: string[]): Promise<Blob[]>
```
* blobIds: Array of blob IDs to download (string[])

### getMedia

```typescript
getMedia(blobId: string): Promise<Blob>
```
* blobId: The ID of a single blob (string)

### Getter/Setter Methods
* `getNetworkUrl()`, `setNetworkUrl(url)`
* `getPackageId()`, `setPackageId(id)`
* `getGasBudget()`, `setGasBudget(budget)`
* `getWalrusNetwork()`, `setWalrusNetwork(network)`
* `getWalrusPublisherUrl()`, `setWalrusPublisherUrl(url)`
* `getWalrusAggregatorUrl()`, `setWalrusAggregatorUrl(url)`
* `getConfig()`

## 🛠️ Types
```typescript
interface OpenGraphClientConfig {
  networkUrl: string;
  packageId: string;
  gasBudget: number;
  walrusNetwork?: string;
  walrusPublisherUrl?: string;
  walrusAggregatorUrl?: string;
}

interface DatasetMetadata {
  name: string;
  description?: string;
  dataType: string;
  dataSize: number;
  creator?: string;
  license?: string;
  tags?: string[];
}

interface WalrusStorageInfo {
  status: WalrusStorageStatus;
  blobId: string;
  endEpoch: number;
  suiRefType: string;
  suiRef: string;
  mediaUrl: string;
  suiScanUrl: string;
  suiRefId: string;
}

enum WalrusStorageStatus {
  NEWLY_CREATED = "NEWLY_CREATED",
  ALREADY_CERTIFIED = "ALREADY_CERTIFIED",
}
```

## 🌐 Walrus URLs

| Network | Publisher URL                               | Aggregator URL                                |
|---------|----------------------------------------------|------------------------------------------------|
| Testnet | https://publisher.testnet.walrus.atalma.io   | https://aggregator.testnet.walrus.atalma.io    |


## License

This project is open-sourced under the ISC License.

You are free to use, copy, modify, and distribute this software with or without fee, provided that the original copyright notice and this permission notice appear.

See the [LICENSE](./LICENSE) file for full license information.


