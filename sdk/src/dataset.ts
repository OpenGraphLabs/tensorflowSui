// src/sdk.ts
import { SuiClient } from "@mysten/sui/client";
import { Transaction } from "@mysten/sui/transactions";
import { OpenGraphClientConfig, DatasetMetadata, WalrusStorageInfo, WalrusStorageStatus } from "./types.js";

export class OpenGraphClient {
  private networkUrl: string;
  private packageId: string;
  private gasBudget: number;
  public suiClient: SuiClient;

  private walrusNetwork = "testnet";
  private walrusPublisherUrl = "https://publisher.testnet.walrus.atalma.io";
  private walrusAggregatorUrl = "https://aggregator.testnet.walrus.atalma.io";

  constructor(config: OpenGraphClientConfig) {
    this.networkUrl = config.networkUrl;
    this.packageId = config.packageId;
    this.gasBudget = config.gasBudget;
    this.suiClient = new SuiClient({ url: config.networkUrl });
    
    this.walrusNetwork = config.walrusNetwork ?? "testnet";
    this.walrusPublisherUrl = config.walrusPublisherUrl ?? "https://publisher.testnet.walrus.atalma.io";
    this.walrusAggregatorUrl = config.walrusAggregatorUrl ?? "https://aggregator.testnet.walrus.atalma.io";
  }

  // =============================
  // Getter Methods
  // =============================
  public getNetworkUrl(): string {
    return this.networkUrl;
  }

  public getPackageId(): string {
    return this.packageId;
  }

  public getGasBudget(): number {
    return this.gasBudget;
  }

  public getWalrusNetwork(): string {
    return this.walrusNetwork;
  }

  public getWalrusPublisherUrl(): string {
    return this.walrusPublisherUrl;
  }

  public getWalrusAggregatorUrl(): string {
    return this.walrusAggregatorUrl;
  }

  public getConfig(): OpenGraphClientConfig {
    return {
      networkUrl: this.networkUrl,
      packageId: this.packageId,
      gasBudget: this.gasBudget,
      walrusNetwork: this.walrusNetwork,
      walrusPublisherUrl: this.walrusPublisherUrl,
      walrusAggregatorUrl: this.walrusAggregatorUrl,
    };
  }

  // =============================
  // Setter Methods
  // =============================
  public setNetworkUrl(url: string): void {
    this.networkUrl = url;
    this.suiClient = new SuiClient({ url }); // 변경 시 client 재생성
  }

  public setPackageId(packageId: string): void {
    this.packageId = packageId;
  }

  public setGasBudget(gasBudget: number): void {
    this.gasBudget = gasBudget;
  }

  public setWalrusNetwork(network: string) {
    this.walrusNetwork = network;
  }

  public setWalrusPublisherUrl(url: string) {
    this.walrusPublisherUrl = url;
  }

  public setWalrusAggregatorUrl(url: string) {
    this.walrusAggregatorUrl = url;
  }

  // Utility
  private getSuiScanUrl(type: "transaction" | "object" | "account", id: string) {
    const baseUrl = `https://suiscan.xyz/${this.walrusNetwork}`;
    if (type === "transaction") {
      return `${baseUrl}/tx/${id}`;
    } else if (type === "account") {
      return `${baseUrl}/account/${id}`;
    } else {
      return `${baseUrl}/object/${id}`;
    }
  }

  // walrus and create ptb tx
  public async uploadDataset(
    files: File[],
    address: string,
    metadata: DatasetMetadata,
    annotations: string[],
    epochs?: number
  ) {
    const storageInfos = await this.uploadDatasetFiles(files, address, epochs);
  
    const dataFiles = await Promise.all(
      storageInfos.map(async (info, i) => {
        const file = files[i];
        const arrayBuffer = await file.arrayBuffer();
        const hashBuffer = await crypto.subtle.digest("SHA-256", arrayBuffer);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        const fileHash = hashArray.map(b => b.toString(16).padStart(2, "0")).join("");
  
        return {
          blobId: info.blobId,
          fileHash,
        };
      })
    );
  
    const tx = await this.createDataset(address, metadata, annotations, dataFiles);
  
    return {
      storageInfos,
      transaction: tx,
    };
  }

  // create ptb
  private async createDataset(
    accountAddress: string,
    metadata: DatasetMetadata,
    annotations: string[],
    files: { blobId: string; fileHash: string }[]
  ) {
    const tx = new Transaction();
    tx.setGasBudget(this.gasBudget);

    const metadataObject = tx.moveCall({
      target: `${this.packageId}::metadata::new_metadata`,
      arguments: [
        tx.pure.option("string", metadata.description),
        tx.pure.string(metadata.dataType),
        tx.pure.u64(BigInt(metadata.dataSize)),
        tx.pure.option("string", metadata.creator),
        tx.pure.option("string", metadata.license),
        tx.pure.option("vector<string>", metadata.tags),
      ],
    });

    const dataset = tx.moveCall({
      target: `${this.packageId}::dataset::new_dataset`,
      arguments: [tx.pure.string(metadata.name), metadataObject],
    });

    for (let i = 0; i < files.length; i++) {
      const rangeOptionObject = tx.moveCall({
        target: `${this.packageId}::dataset::new_range_option`,
        arguments: [tx.pure.option("u64", null), tx.pure.option("u64", null)],
      });

      const dataObject = tx.moveCall({
        target: `${this.packageId}::dataset::new_data`,
        arguments: [
          tx.pure.string(`data_${i}`),
          tx.pure.string(files[i].blobId),
          tx.pure.string(files[i].fileHash),
          rangeOptionObject,
        ],
      });

      tx.moveCall({
        target: `${this.packageId}::dataset::add_annotation_label`,
        arguments: [dataObject, tx.pure.string(annotations[i])],
      });

      tx.moveCall({
        target: `${this.packageId}::dataset::add_data`,
        arguments: [dataset, dataObject],
      });
    }

    tx.transferObjects([dataset], accountAddress);
    return tx;
  }

  public async getDatasets(ownerAddress: string) {
    if (!ownerAddress) {
      throw new Error("OwnerAddress ID is required");
    }

    const { data } = await this.suiClient.getOwnedObjects({ owner: ownerAddress });
    const objectIds = data
      .map(d => d.data?.objectId)
      .filter((id): id is string => id !== undefined);

    const objects = await this.suiClient.multiGetObjects({
      ids: objectIds,
      options: { showContent: true, showType: true },
    });

    return objects
      .filter(obj =>
        obj.data?.content?.dataType === "moveObject" &&
        obj.data?.content?.type?.includes("dataset::Dataset"))
      .map(obj => {
        const content = obj.data?.content as any;
        return {
          id: obj.data?.objectId,
          name: content.fields.name,
          description: content.fields.description,
          dataType: content.fields.data_type,
          dataSize: content.fields.data_size,
          creator: content.fields.creator,
          license: content.fields.license,
          tags: content.fields.tags,
        };
      });
  }

  public async getDatasetById(datasetId: string) {
    if (!datasetId) {
      throw new Error("Dataset ID is required");
    }

    const object = await this.suiClient.getObject({
      id: datasetId,
      options: { showContent: true, showType: true },
    });

    if (object.data?.content?.dataType !== "moveObject") {
      throw new Error("Invalid dataset object");
    }

    const content = object.data?.content as any;
    return {
      id: object.data?.objectId,
      name: content.fields.name,
      description: content.fields.description,
      dataType: content.fields.data_type,
      dataSize: content.fields.data_size,
      creator: content.fields.creator,
      license: content.fields.license,
      tags: content.fields.tags,
    };
  }

  // walrus
  private async uploadDatasetFiles(files: File[], address: string, epochs?: number): Promise<WalrusStorageInfo[]> {
    const uploadPromises = files.map(file => this.uploadMedia(file, address, epochs));
    return await Promise.all(uploadPromises);
  }

  // 미디어 업로드
  private async uploadMedia(file: File, sendTo: string, epochs?: number): Promise<WalrusStorageInfo> {
    try {
      let epochsParam = epochs ? `&epochs=${epochs}` : "";
      const url = `${this.walrusPublisherUrl}/v1/blobs?send_object_to=${sendTo}${epochsParam}`;

      const response = await fetch(url, {
        method: "PUT",
        headers: {
          "Content-Type": file.type,
          "Content-Length": file.size.toString(),
        },
        body: file,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`업로드 실패: ${response.status} ${response.statusText} - ${errorText}`);
      }

      const data = await response.json();
      let storageInfo: WalrusStorageInfo;

      if ("alreadyCertified" in data) {
        storageInfo = {
          status: WalrusStorageStatus.ALREADY_CERTIFIED,
          blobId: data.alreadyCertified.blobId,
          endEpoch: data.alreadyCertified.endEpoch,
          suiRefType: "Previous Sui Certified Event",
          suiRef: data.alreadyCertified.event.txDigest,
          mediaUrl: `${this.walrusAggregatorUrl}/v1/blobs/${data.alreadyCertified.blobId}`,
          suiScanUrl: this.getSuiScanUrl("transaction", data.alreadyCertified.event.txDigest),
          suiRefId: data.alreadyCertified.event.txDigest,
        };
      } else if ("newlyCreated" in data) {
        storageInfo = {
          status: WalrusStorageStatus.NEWLY_CREATED,
          blobId: data.newlyCreated.blobObject.blobId,
          endEpoch: data.newlyCreated.blobObject.storage.endEpoch,
          suiRefType: "Associated Sui Object",
          suiRef: data.newlyCreated.blobObject.id,
          mediaUrl: `${this.walrusAggregatorUrl}/v1/blobs/${data.newlyCreated.blobObject.blobId}`,
          suiScanUrl: this.getSuiScanUrl("object", data.newlyCreated.blobObject.id),
          suiRefId: data.newlyCreated.blobObject.id,
        };
      } else {
        throw new Error("알 수 없는 응답 형식");
      }

      return storageInfo;
    } catch (error) {
      console.error("미디어 업로드 오류:", error);
      throw new Error(
        `미디어 업로드 실패: ${error instanceof Error ? error.message : "알 수 없는 오류"}`
      );
    }
  }

  public async uploadTrainingData(file: File, address: string): Promise<WalrusStorageInfo> {
    const response = await fetch(`${this.walrusPublisherUrl}/v1/blobs?send_object_to=${address}`, {
      method: "PUT",
      body: file,
    });

    if (!response.ok) {
      throw new Error(`Failed to upload file: ${response.statusText}`);
    }

    const data = await response.json();
    return this.transformResponse(data);
  }

  private transformResponse(data: any): WalrusStorageInfo {
    if ("alreadyCertified" in data) {
      return {
        status: WalrusStorageStatus.ALREADY_CERTIFIED,
        blobId: data.alreadyCertified.blobId,
        endEpoch: data.alreadyCertified.endEpoch,
        suiRefType: "Previous Sui Certified Event",
        suiRef: data.alreadyCertified.event.txDigest,
        mediaUrl: `${this.walrusAggregatorUrl}/v1/blobs/${data.alreadyCertified.blobId}`,
        suiScanUrl: this.getSuiScanUrl("transaction", data.alreadyCertified.event.txDigest),
        suiRefId: data.alreadyCertified.event.txDigest,
      };
    } else if ("newlyCreated" in data) {
      return {
        status: WalrusStorageStatus.NEWLY_CREATED,
        blobId: data.newlyCreated.blobObject.blobId,
        endEpoch: data.newlyCreated.blobObject.storage.endEpoch,
        suiRefType: "Associated Sui Object",
        suiRef: data.newlyCreated.blobObject.id,
        mediaUrl: `${this.walrusAggregatorUrl}/v1/blobs/${data.newlyCreated.blobObject.blobId}`,
        suiScanUrl: this.getSuiScanUrl("object", data.newlyCreated.blobObject.id),
        suiRefId: data.newlyCreated.blobObject.id,
      };
    } else {
      throw new Error("Unknown response format");
    }
  }

  public async getTrainingData(blobIds: string[]): Promise<Blob[]> {
    try {
      const getPromises = blobIds.map(blobId => this.getMedia(blobId));
      return await Promise.all(getPromises);
    } catch (error) {
      console.error("학습 데이터 가져오기 오류:", error);
      throw error;
    }
  }

  public async getMedia(blobId: string): Promise<Blob> {
    try {
      const url = `${this.walrusAggregatorUrl}/v1/blobs/${blobId}`;
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`미디어 가져오기 실패: ${response.status} ${response.statusText}`);
      }

      return await response.blob();
    } catch (error) {
      console.error("미디어 가져오기 오류:", error);
      throw error;
    }
  }
}