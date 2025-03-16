# TensorflowSui v.2.0.1

TensorflowSui is a project that enables the deployment and inference of machine learning models fully onchain on the Sui blockchain. This project implements Tensorflow for Web3 Sui, making fully onchain inference possible.

## Key Features

- Onchain neural network model deployment
- Fully onchain inference
- Custom model support (new in v.2.0.1)

## Module Structure

TensorflowSui consists of the following key modules:

1. **tensor**: Defines tensor data structures and operations.
2. **graph**: Defines neural network graph structures and layers.
3. **model**: Provides model creation and initialization functionality.

## v.2.0.1 New Feature: Custom Models

Version 2.0.1 introduces the ability for users to create custom models by providing their own model data. Previous versions had hardcoded MNIST models with limited extensibility, but now various models can be deployed onchain.

### Creating Custom Models

You can create custom models using the `initialize_model` function:

```bash
sui client call \
  --package {tensorflowSui_package_id} \
  --module model \
  --function initialize_model \
  --args \
    "[\"layer1\", \"layer2\"]" \
    "[4, 3]" \
    "[3, 2]" \
    "[[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 12], [15, 15, 15, 15, 15, 6]]" \
    "[[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1]]" \
    "[[5, 5, 5], [7, 7]]" \
    "[[0, 0, 0], [0, 0]]" \
    "2" 
```
