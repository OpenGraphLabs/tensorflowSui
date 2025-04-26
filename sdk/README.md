OpenGraphClient SDK
Easily upload datasets and manage Sui blockchain assets via Walrus Publisher and Aggregator.

📦 Installation
bash
npm install YOUR_PACKAGE_NAME
or

bash
복사
편집
yarn add YOUR_PACKAGE_NAME
🚀 Quick Start
typescript
복사
편집
import { OpenGraphClient } from "your-package-name";

// Initialize the client
const client = new OpenGraphClient({
  networkUrl: "https://fullnode.testnet.sui.io",
  packageId: "0x...",
  gasBudget: 100000000,
});

// Upload and register dataset
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
📚 API Overview
Constructor
typescript
복사
편집
new OpenGraphClient(config: OpenGraphClientConfig);

Param	Type	Description
networkUrl	string	Sui RPC endpoint URL
packageId	string	Smart contract package ID
gasBudget	number	Default gas budget
walrusNetwork?	string	Walrus network (default: testnet)
walrusPublisherUrl?	string	Walrus publisher API URL
walrusAggregatorUrl?	string	Walrus aggregator API URL
Main Methods
uploadDataset(files, address, metadata, annotations, epochs?)
Uploads files to Walrus and prepares a transaction for dataset creation.

files: File[]

address: string

metadata: DatasetMetadata

annotations: string[]

epochs?: number (optional)

Returns:

typescript
복사
편집
{
  storageInfos: WalrusStorageInfo[],
  transaction: Transaction
}
getDatasets(ownerAddress)
Fetch all datasets owned by a specific Sui address.

ownerAddress: string

Returns an array of dataset metadata.

getDatasetById(datasetId)
Fetch a single dataset by its object ID.

datasetId: string

Returns a single dataset metadata object.

uploadTrainingData(file, address)
Upload a single file intended for training.

file: File

address: string

Returns a WalrusStorageInfo object.

getTrainingData(blobIds)
Download multiple blobs by their blob IDs.

blobIds: string[]

Returns a Blob[].

getMedia(blobId)
Download a single blob by its blob ID.

blobId: string

Returns a Blob.

Getter/Setter Methods
getNetworkUrl(), setNetworkUrl(url)

getPackageId(), setPackageId(id)

getGasBudget(), setGasBudget(budget)

getWalrusNetwork(), setWalrusNetwork(network)

getWalrusPublisherUrl(), setWalrusPublisherUrl(url)

getWalrusAggregatorUrl(), setWalrusAggregatorUrl(url)

getConfig()

🛠️ Types
typescript
복사
편집
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
🌐 Walrus URLs

Network	Publisher URL	Aggregator URL
Testnet	https://publisher.testnet.walrus.atalma.io	https://aggregator.testnet.walrus.atalma.io
You can override these URLs using the constructor options.

📄 License
MIT License