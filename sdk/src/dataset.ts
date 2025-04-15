// src/dataset.ts
import { SuiClient } from "@mysten/sui/client";
import { Transaction } from "@mysten/sui/transactions";
import {
  getSuiNetwork,
  getContractConfig,
  getGasBudget,
} from "./config.js";
import { DatasetMetadata } from "./types.js";

const suiClient = new SuiClient({
  url: getSuiNetwork().URL,
});

export async function createDataset(
  accountAddress: string,
  metadata: DatasetMetadata,
  annotations: string[],
  files: { blobId: string; fileHash: string }[]
) {
  const contract = getContractConfig();
  const tx = new Transaction();
  tx.setGasBudget(getGasBudget());

  const metadataObject = tx.moveCall({
    target: `${contract.PACKAGE_ID}::metadata::new_metadata`,
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
    target: `${contract.PACKAGE_ID}::dataset::new_dataset`,
    arguments: [tx.pure.string(metadata.name), metadataObject],
  });

  for (let i = 0; i < files.length; i++) {
    const rangeOptionObject = tx.moveCall({
      target: `${contract.PACKAGE_ID}::dataset::new_range_option`,
      arguments: [tx.pure.option("u64", null), tx.pure.option("u64", null)],
    });

    const dataObject = tx.moveCall({
      target: `${contract.PACKAGE_ID}::dataset::new_data`,
      arguments: [
        tx.pure.string(`data_${i}`),
        tx.pure.string(files[i].blobId),
        tx.pure.string(files[i].fileHash),
        rangeOptionObject,
      ],
    });

    tx.moveCall({
      target: `${contract.PACKAGE_ID}::dataset::add_annotation_label`,
      arguments: [dataObject, tx.pure.string(annotations[i])],
    });

    tx.moveCall({
      target: `${contract.PACKAGE_ID}::dataset::add_data`,
      arguments: [dataset, dataObject],
    });
  }

  tx.transferObjects([dataset], accountAddress);

  return tx;
}

export async function getDatasets(ownerAddress: string) {
  if (!ownerAddress) {
    return {
      datasets: [],
      isLoading: false,
      error: null,
    };
  }

  try {
    // 해당 주소가 소유한 객체 가져오기
    const { data } = await suiClient.getOwnedObjects({
      owner: ownerAddress,
    });

    // 객체 ID 추출
    const objectIds = data
      .map(item => item.data?.objectId)
      .filter((id): id is string => id !== undefined);

    if (!objectIds.length) {
      return {
        datasets: [],
        isLoading: false,
        error: null,
      };
    }

    // 객체 상세 정보 가져오기
    const objects = await suiClient.multiGetObjects({
      ids: objectIds,
      options: {
        showContent: true,
        showType: true,
      },
    });

    // 데이터셋 객체만 필터링
    const datasetObjects = objects.filter(
      obj =>
        obj.data?.content?.dataType === "moveObject" &&
        obj.data?.content?.type?.includes("dataset::Dataset")
    );

    // 데이터셋 객체 파싱
    const datasets = datasetObjects.map(obj => {
      const content = obj.data?.content as any;
      return {
        id: obj.data?.objectId,
        name: content?.fields?.name,
        description: content?.fields?.description,
        dataType: content?.fields?.data_type,
        dataSize: content?.fields?.data_size,
        creator: content?.fields?.creator,
        license: content?.fields?.license,
        tags: content?.fields?.tags,
      };
    });

    return {
      datasets,
      isLoading: false,
      error: null,
    };
  } catch (error) {
    console.error("Error fetching datasets:", error);
    return {
      datasets: [],
      isLoading: false,
      error,
    };
  }
}

export async function getDatasetById(datasetId: string) {
  if (!datasetId) {
    throw new Error("Dataset ID is required");
  }

  try {
    const object = await suiClient.getObject({
      id: datasetId,
      options: {
        showContent: true,
        showType: true,
      },
    });

    if (object.data?.content?.dataType !== "moveObject") {
      throw new Error("Invalid dataset object");
    }

    const content = object.data.content as any;
    return {
      dataset: {
        id: object.data.objectId,
        name: content.fields?.name,
        description: content.fields?.description,
        dataType: content.fields?.data_type,
        dataSize: content.fields?.data_size,
        creator: content.fields?.creator,
        license: content.fields?.license,
        tags: content.fields?.tags,
      },
      isLoading: false,
      error: null,
    };
  } catch (error) {
    console.error("Error fetching dataset by ID:", error);
    throw error;
  }
}
