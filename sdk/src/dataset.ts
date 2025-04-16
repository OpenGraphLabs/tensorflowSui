// src/sdk.ts
import { SuiClient } from "@mysten/sui/client";
import { Transaction } from "@mysten/sui/transactions";
import type { DatasetSDKConfig, DatasetMetadata } from "./types.js";

export class DatasetSDK {
  private networkUrl: string;
  private packageId: string;
  private gasBudget: number;
  public suiClient: SuiClient;

  constructor(config: DatasetSDKConfig) {
    this.networkUrl = config.networkUrl;
    this.packageId = config.packageId;
    this.gasBudget = config.gasBudget;

    this.suiClient = new SuiClient({ url: this.networkUrl });
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

  public getConfig(): DatasetSDKConfig {
    return {
      networkUrl: this.networkUrl,
      packageId: this.packageId,
      gasBudget: this.gasBudget,
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

  public async createDataset(
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
}