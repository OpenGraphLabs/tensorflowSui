import h5py
import numpy as np
import json
import os
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Any
import tensorflow as tf

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

def load_h5_model(model_path: str) -> Dict[str, Any]:
    """Load and process H5 model weights."""
    model = tf.keras.models.load_model(model_path)
    
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
    
    return converted_weights

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

def main():
    """Main execution function."""
    print_banner("O P E N G R A P H")
    
    # Load configuration
    config = load_config()
    model_path = config["H5_MODEL_PATH"]
    scale = int(config["SCALE"])
    
    # Process model
    print("\n1. Processing H5 model...")
    weights = load_h5_model(model_path)
    converted_weights = convert_weights_to_fixed(weights, scale)
    
    # Generate Move code
    move_code = generate_move_code(converted_weights, scale)
    
    # Create directory if it doesn't exist
    os.makedirs('./with_git_dependencies/sources', exist_ok=True)
    
    # Generate and save Move.toml
    move_toml = generate_move_toml(config)
    with open('./with_git_dependencies/Move.toml', 'w') as f:
        f.write(move_toml)
    print("Move.toml generated and saved to ./with_git_dependencies/Move.toml")
    
    # Save Move code
    with open('./with_git_dependencies/sources/model.move', 'w') as f:
        f.write(move_code)
    print("Move code generated and saved to ./with_git_dependencies/sources/model.move")

if __name__ == "__main__":
    main()
