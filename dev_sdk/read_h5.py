# Suppress TensorFlow warnings
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logging

import h5py
import numpy as np
import json
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Any
import tensorflow as tf
import subprocess
from pysui import SuiConfig, SyncClient
from pysui.sui.sui_builders.exec_builders import ExecuteTransaction
from pysui.sui.sui_types.address import SuiAddress

def print_banner(text: str) -> None:
    """Display ASCII art banner with gradient colors."""
    letters = {
        "O": [" ███ ", "█   █", "█   █", "█   █", " ███ "],
        "P": ["████ ", "█   █", "████ ", "█    ", "█    "],
        "E": ["████", "█   ", "███ ", "█   ", "████"],
        "N": ["█   █", "██  █", "█ █ █", "█  ██", "█   █"],
        "G": [" ███ ", "█    ", "█  ██", "█   █", " ███ "],
        "R": ["████ ", "█   █", "████ ", "█  █ ", "█   █"],
        "A": [" ███ ", "█   █", "█████", "█   █", "█   █"],
        "H": ["█   █", "█   █", "█████", "█   █", "█   █"]
    }
    
    output = ["", "", "", "", ""]
    colors = [
        "\033[38;5;51m",  # Cyan
        "\033[38;5;45m",  # Light Blue
        "\033[38;5;39m",  # Blue
        "\033[38;5;33m",  # Darker Blue
        "\033[38;5;27m"   # Deep Blue
    ]
    reset = "\033[0m"

    for char in text:
        if char in letters:
            for i, line in enumerate(letters[char]):
                output[i] += f"{colors[i]}{line}{reset}   "
        elif char == " ":
            for i in range(5):
                output[i] += "  "

    print("\n" + "\n".join(output) + "\n\n")

def load_config() -> Dict[str, Any]:
    """Load configuration from config.txt file."""
    config = {}
    with open('../config.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = map(str.strip, line.split('='))
                    config[key] = value.strip("'\" \t;")
    return config

def float_to_fixed(x: float, scale: int) -> Tuple[int, int]:
    """Convert floating point to fixed point representation."""
    sign_bit = 1 if x < 0 else 0
    x = abs(x)
    factor = 10 ** scale
    abs_val = round(x * factor)
    return sign_bit, abs_val

def load_h5_model(model_path: str) -> List[Dict]:
    """Step 1: Load and process H5 model weights."""
    print("\n1. Loading H5 model from:", model_path)
    
    try:
        # Custom object scope to handle the reduction error
        custom_objects = {
            'reduction': 'sum_over_batch_size'  # Set default reduction method
        }
        
        # Load model with custom objects
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False  # Skip compilation to avoid metric/loss issues
        )
        
        converted_weights = []
        for layer in model.layers:
            if not layer.weights:  # Skip layers without weights
                continue
                
            weights = layer.get_weights()
            if len(weights) != 2:  # Skip layers without kernel and bias
                continue
                
            kernel, bias = weights
            converted_weights.append({
                "layerName": layer.name,
                "kernel": {
                    "shape": list(kernel.shape),
                    "data": kernel
                },
                "bias": {
                    "shape": list(bias.shape),
                    "data": bias
                }
            })
        
        print(f"Successfully loaded model with {len(converted_weights)} layers")
        return converted_weights
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # Alternative loading method using h5py directly
        try:
            print("Attempting to load weights directly using h5py...")
            with h5py.File(model_path, 'r') as f:
                converted_weights = []
                # Iterate through model layers
                for layer_name in f.keys():
                    if 'dense' in layer_name:
                        weights = f[layer_name]['dense']['kernel:0'][()]
                        bias = f[layer_name]['dense']['bias:0'][()]
                        
                        converted_weights.append({
                            "layerName": layer_name,
                            "kernel": {
                                "shape": list(weights.shape),
                                "data": weights
                            },
                            "bias": {
                                "shape": list(bias.shape),
                                "data": bias
                            }
                        })
                
                print(f"Successfully loaded model with {len(converted_weights)} layers using h5py")
                return converted_weights
                
        except Exception as e2:
            print(f"Error loading model with h5py: {str(e2)}")
            raise Exception("Failed to load model using both methods") from e2

def convert_weights_to_fixed(weights: List[Dict], scale: int) -> List[Dict]:
    """Convert weights to fixed-point representation."""
    converted = []
    
    for layer in weights:
        kernel_signs = []
        kernel_mags = []
        for val in layer["kernel"]["data"].flatten():
            sign, mag = float_to_fixed(float(val), scale)
            kernel_signs.append(sign)
            kernel_mags.append(mag)
            
        bias_signs = []
        bias_mags = []
        for val in layer["bias"]["data"].flatten():
            sign, mag = float_to_fixed(float(val), scale)
            bias_signs.append(sign)
            bias_mags.append(mag)
            
        converted.append({
            "layerName": layer["layerName"],
            "kernel": {
                "magnitude": kernel_mags,
                "sign": kernel_signs,
                "shape": layer["kernel"]["shape"]
            },
            "bias": {
                "magnitude": bias_mags,
                "sign": bias_signs,
                "shape": layer["bias"]["shape"]
            },
            "scale": scale
        })
    
    return converted

def generate_move_code(converted_weights: List[Dict], scale: int) -> str:
    """Generate Move smart contract code."""
    move_code = f"""module models::model {{
    use sui::tx_context::TxContext;
    use tensorflowsui::graph;
    use tensorflowsui::tensor;

    public fun create_model_signed_fixed(graph: &mut graph::SignedFixedGraph, scale: u64) {{
"""
    
    # Add layer declarations
    for layer in converted_weights:
        input_size, output_size = layer["kernel"]["shape"]
        move_code += f'        graph::DenseSignedFixed(graph, {input_size}, {output_size}, b"{layer["layerName"]}", scale);\n'
    
    # Add weights for each layer
    for layer in converted_weights:
        kernel_mag = f"vector[{', '.join(map(str, layer['kernel']['magnitude']))}]"
        kernel_sign = f"vector[{', '.join(map(str, layer['kernel']['sign']))}]"
        bias_mag = f"vector[{', '.join(map(str, layer['bias']['magnitude']))}]"
        bias_sign = f"vector[{', '.join(map(str, layer['bias']['sign']))}]"
        
        move_code += f"""
        let w{layer['layerName']}_mag = {kernel_mag};
        let w{layer['layerName']}_sign = {kernel_sign};
        let b{layer['layerName']}_mag = {bias_mag};
        let b{layer['layerName']}_sign = {bias_sign};

        graph::set_layer_weights_signed_fixed(
            graph,
            b"{layer['layerName']}",
            w{layer['layerName']}_mag, w{layer['layerName']}_sign,
            b{layer['layerName']}_mag, b{layer['layerName']}_sign,
            {layer['kernel']['shape'][0]}, {layer['kernel']['shape'][1]},
            scale
        );"""
    
    # Add helper functions
    move_code += """
    }

    entry public fun split_chunk_compute(
        graph_obj: &graph::SignedFixedGraph,
        pd: &mut graph::PartialDenses,
        partial_name: vector<u8>,
        input_magnitude: vector<u64>, input_sign: vector<u64>,
        activation_type: u64,
        start_j: u64,
        end_j: u64
    ) {
        graph::split_chunk_compute(graph_obj, pd, partial_name, input_magnitude, input_sign, activation_type, start_j, end_j);    
    }

    entry public fun split_chunk_finalize(
        pd: &mut graph::PartialDenses,
        partial_name: vector<u8>
    ): (vector<u64>, vector<u64>, u64) {
        let mut results_mag = vector::empty<u64>();
        let mut result_sign = vector::empty<u64>();
        let mut results;
        let (_results_mag, _result_sign, _results) = graph::split_chunk_finalize(pd, partial_name);

        results_mag = _results_mag;
        result_sign = _result_sign;
        results = _results;

        (results_mag, result_sign, results)
    }

    entry public fun ptb_layer(
        graph: &graph::SignedFixedGraph,
        input_magnitude: vector<u64>, input_sign: vector<u64>,
        scale: u64, name: vector<u8>
    ) : (vector<u64>, vector<u64>, u64) {
        let mut results_mag = vector::empty<u64>();
        let mut result_sign = vector::empty<u64>();
        let mut results;

        let (_results_mag, _result_sign, _results) = graph::ptb_layer(graph, input_magnitude, input_sign, scale, name);

        results_mag = _results_mag;
        result_sign = _result_sign;
        results = _results;

        (results_mag, result_sign, results)
    }

    entry public fun ptb_layer_arg_max(
        graph: &graph::SignedFixedGraph,
        input_magnitude: vector<u64>, input_sign: vector<u64>,
        scale: u64, name: vector<u8>
    ) : u64 {
        let results = graph::ptb_layer_arg_max(graph, input_magnitude, input_sign, scale, name);
        results
    }

    public entry fun initialize(ctx: &mut TxContext) {
        let mut graph = graph::create_signed_graph(ctx);
        create_model_signed_fixed(&mut graph, {scale}); 
        let mut partials = graph::create_partial_denses(ctx);
        graph::add_partials_for_all_but_last(&graph, &mut partials);
        graph::share_graph(graph);
        graph::share_partial(partials);
    }
}}"""
    
    return move_code

def generate_move_toml(config: Dict[str, Any]) -> str:
    """Generate Move.toml file content."""
    return f"""[package]
name = "Model"
edition = "2024.beta"

[dependencies]
Sui = {{ git = "https://github.com/MystenLabs/sui.git", subdir = "crates/sui-framework/packages/sui-framework", rev = "framework/{config['NETWORK']}" }}
tensorflowsui = {{ git = "https://github.com/depinity/tensorflowsui.git", subdir = "tensorflowSuiLib/v.1.0.1", rev = "main" }}

[addresses]
models = "0x0"

[dev-dependencies]

[dev-addresses]
"""

def generate_move_files(converted_weights: List[Dict], scale: int, config: Dict[str, Any]) -> None:
    """Step 2 & 3: Generate Move code and Move.toml."""
    print("\n2. Generating Move files...")
    
    # Create directory if it doesn't exist
    os.makedirs('./with_git_dependencies/sources', exist_ok=True)
    
    # Generate and save Move.toml
    move_toml = generate_move_toml(config)
    with open('./with_git_dependencies/Move.toml', 'w') as f:
        f.write(move_toml)
    print("Move.toml generated and saved")
    
    # Generate and save model.move
    move_code = generate_move_code(converted_weights, scale)
    with open('./with_git_dependencies/sources/model.move', 'w') as f:
        f.write(move_code)
    print("model.move generated and saved")

def publish_to_network(config: Dict[str, Any]) -> Tuple[str, str]:
    """Step 4: Publish to SUI network using pysui."""
    print("\n3. Publishing to " + config['NETWORK'] + "...")
    
    # Initialize Sui configuration
    sui_config = SuiConfig.user_config(
        rpc_url=config['RPC_URL'],
        prv_keys=[config['PRIVATE_KEY']]
    )
    
    # Create sync client
    client = SyncClient(sui_config)
    
    # Build the Move package
    try:
        build_output = subprocess.check_output(
            "sui move build --dump-bytecode-as-base64 --path ./with_git_dependencies --silence-warnings",
            shell=True,
            text=True
        )
        build_data = json.loads(build_output)
        print("Build successful!")
        
        # Create execute transaction builder
        execute_txn = ExecuteTransaction(
            client=client,
            modules=build_data['modules'],
            dependencies=build_data['dependencies'],
            gas_budget=config.get('GAS_BUDGET', 50000000)
        )
        
        # Execute transaction
        result = client.execute(execute_txn)
        
        if result.is_ok():
            # Extract package ID from result
            package_id = result.result_data.created[0].reference.object_id
            
            print("\nPackage ID:", package_id)
            print(f"https://suiscan.xyz/{config['NETWORK']}/object/{package_id}/tx-blocks")
            
            print("\nTransaction Digest:", result.digest)
            print(f"https://suiscan.xyz/{config['NETWORK']}/tx/{result.digest}")
            
            # Save package ID
            with open('../packageId.txt', 'w') as f:
                f.write(package_id)
            print("Package ID saved to packageId.txt")
            
            return package_id, result.digest
        else:
            print("Transaction failed:", result)
            raise Exception("Deployment failed")
            
    except Exception as e:
        print("Error executing transaction:", e)
        raise

def store_train_data(package_id: str, digest: str, config: Dict[str, Any]) -> str:
    """Store training data in Walrus."""
    try:
        # Load training and test data
        with open('./web2_datasets/resample_convert_train.json', 'r') as f:
            train_data = json.load(f)
        with open('./web2_datasets/resample_convert_test.json', 'r') as f:
            test_data = json.load(f)
            
        print('MNIST data loaded successfully')
        
        # Send request to Walrus server
        payload = {
            "train": train_data,
            "test": test_data,
            "packageId": package_id,
            "digest": digest
        }
        
        response = requests.post(
            'http://localhost:8083/train-set',
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        data = response.json()
        blob_id = data['blobId']
        
        print("Training Set Walrus Store Success", data)
        print(f'https://walruscan.com/testnet/blob/{blob_id}')
        
        return blob_id
    except Exception as e:
        print('API Call Error:', e)
        return ""

def main():
    """Main execution function."""
    print_banner("O P E N G R A P H")
    
    # Load configuration
    config = load_config()
    model_path = config["H5_MODEL_PATH"]
    scale = int(config["SCALE"])
    
    # Step 1: Load and process H5 model
    weights = load_h5_model(model_path)
    converted_weights = convert_weights_to_fixed(weights, scale)
    
    # Step 2 & 3: Generate Move files
    generate_move_files(converted_weights, scale, config)
    
    # Step 4: Publish to network
    package_id, digest = publish_to_network(config)
    
    # Store training data
    blob_id = store_train_data(package_id, digest, config)
    
    print("\nDeployment completed successfully!")

if __name__ == "__main__":
    main()
