# TensorflowSui v.2.0.1

TensorflowSui is a project that enables the deployment and inference of machine learning models fully onchain on the Sui blockchain. This project implements Tensorflow for Web3 Sui, making fully onchain inference possible.

## Key Features

- Onchain neural network model deployment
- Fully onchain inference
- Custom model support (new in v.2.0.1)
- Model metadata as String type (improved readability)
- Model metadata support (name, description, task type)
- Tensorflow-style prediction API (predict and predict_class functions)
- Layer-by-layer inference for gas efficiency

## Module Structure

TensorflowSui consists of the following key modules:

1. **tensor**: Defines tensor data structures and operations.
2. **graph**: Defines neural network graph structures and layers.
3. **model**: Provides model creation and initialization functionality.

## v.2.0.1 New Feature: Custom Models

Version 2.0.1 introduces the ability for users to create custom models by providing their own model data. Previous versions had hardcoded MNIST models with limited extensibility, but now various models can be deployed onchain.

### Creating Custom Models

You can create custom models using the `create_model` function:

```bash
sui client call \
  --package {tensorflowSui_package_id} \
  --module model \
  --function create_model \
  --args \
    "\"My Custom Model\"" \
    "\"A neural network for digit recognition\"" \
    "\"classification\"" \
    "[[4, 3], [3, 2]]" \
    "[[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 12], [15, 15, 15, 15, 15, 6]]" \
    "[[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1]]" \
    "[[5, 5, 5], [7, 7]]" \
    "[[0, 0, 0], [0, 0]]" \
    "2" 
```

### Running Inference with the Model

TensorflowSui v.2.0.1 provides several options for model inference:

1. **predict()**: Full inference function that returns magnitude and sign vectors along with the argmax index
2. **predict_class()**: Simplified function that returns only the predicted class index (for classification tasks)
3. **predict_layer()**: Processes a single layer for gas-efficient computation

#### Using the predict() function:

```bash
sui client call \
  --package {tensorflowSui_package_id} \
  --module model \
  --function predict \
  --args \
    "{model_object_id}" \
    "[1, 2, 3, 4, 1, 2, 3, 4, 5, 6]" \
    "[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]" 
```

The predict function returns a tuple of three values:
- Magnitude vector of the output
- Sign vector of the output (0 for positive, 1 for negative)
- Argmax index (the predicted class for classification models)

#### Using the predict_class() function:

```bash
sui client call \
  --package {tensorflowSui_package_id} \
  --module model \
  --function predict_class \
  --args \
    "{model_object_id}" \
    "[1, 2, 3, 4]" \
    "[0, 0, 0, 0]" 
```

The predict_class function returns just the index of the maximum value in the output, which corresponds to the predicted class for classification models.

### Gas-Efficient Layer-by-Layer Inference

For large models, performing the entire inference in a single transaction may be too gas-intensive. TensorflowSui v.2.0.1 provides the `predict_layer()` function for layer-by-layer inference:

#### Using predict_layer():

This approach processes a single layer and emits an event with the output. This is useful for large models where processing all layers at once would be too gas-intensive.

```bash
sui client call \
  --package {tensorflowSui_package_id} \
  --module model \
  --function predict_layer \
  --args \
    "{model_object_id}" \
    "0" \                # Layer index (starting at 0)
    "[1, 2, 3, 4, 1, 2, 3, 4, 5, 6]" \
    "[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]" 
```

The function returns the layer's output as a tuple of (magnitude_vector, sign_vector, optional_argmax). For non-final layers, the third value is None. For the final layer, it contains the argmax index and also automatically emits a `PredictionCompleted` event. 

The output can then be used as input to the next layer in a subsequent transaction:

```bash
# Process the second layer using the output from the first layer
sui client call \
  --package {tensorflowSui_package_id} \
  --module model \
  --function predict_layer \
  --args \
    "{model_object_id}" \
    "1" \                # Layer index (now processing layer 1)
    "[22, 28]" \         # Output from previous layer
    "[0, 0]"             # Signs from previous layer (all positive)
```

When processing the final layer (when layer_idx is the last layer), this function automatically:
1. Calculates the argmax value
2. Returns it as part of the tuple
3. Emits a `PredictionCompleted` event with the final prediction

For non-final layers, it emits a `LayerComputed` event, allowing clients to track the intermediate results through events.

### Model Schema

The `create_model` function accepts parameters that match the following schema:

```json
{
  "name": "My Custom Model",
  "description": "A neural network for digit recognition",
  "taskType": "classification",
  "layerDimensions": [
    [4, 3],  # 첫 번째 레이어: 입력 4, 출력 3
    [3, 2]   # 두 번째 레이어: 입력 3, 출력 2
  ],
  "weightsMagnitudes": [
    [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 12],
    [15, 15, 15, 15, 15, 6]
  ],
  "weightsSigns": [
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 1]
  ],
  "biasesMagnitudes": [
    [5, 5, 5],
    [7, 7]
  ],
  "biasesSigns": [
    [0, 0, 0],
    [0, 0]
  ],
  "scale": 2
}
```

Where:
- `name`: Name of the model (stored as String)
- `description`: Detailed description of the model (stored as String)
- `taskType`: Type of task the model performs (e.g., "classification", "regression") (stored as String)
- `layerDimensions`: A list of [input_dimension, output_dimension] pairs for each layer
- `weightsMagnitudes`: Magnitude values for weights in each layer
- `weightsSigns`: Sign values (0 for positive, 1 for negative) for weights in each layer
- `biasesMagnitudes`: Magnitude values for biases in each layer
- `biasesSigns`: Sign values (0 for positive, 1 for negative) for biases in each layer
- `scale`: Fixed point scale factor (2^scale)

### String Type Metadata

In this version, we've enhanced the model metadata by using Sui's native String type instead of raw byte vectors. This provides several benefits:

1. **Better Readability**: Strings are displayed properly in explorers and other tools
2. **UTF-8 Support**: Full support for international characters and symbols
3. **Easier Access**: Helper functions like `get_name()`, `get_description()`, and `get_task_type()` return properly formatted strings
4. **Improved Indexing**: Better indexing and searching capabilities in the blockchain explorer

This change makes it easier to work with model metadata both on-chain and in client applications.
