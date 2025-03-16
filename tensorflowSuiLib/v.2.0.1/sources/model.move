// Copyright (c) OpenGraph, Inc.
// SPDX-License-Identifier: Apache-2.0

/// @title Fully Onchain Neural Network Inference Implementation
module tensorflowsui::model {
    use tensorflowsui::graph::{Self, SignedFixedGraph};
    
    /// @dev Error when input dimension does not match
    const EInputDimMismatch: u64 = 1002;
    /// @dev Error when output dimension does not match
    const EOutputDimMismatch: u64 = 1003;
    /// @dev Error when weight magnitude vector does not match
    const EWeightsMagnitudeMismatch: u64 = 1004;
    /// @dev Error when weight sign vector does not match
    const EWeightsSignMismatch: u64 = 1005;
    /// @dev Error when bias magnitude vector does not match
    const EBiasesMagnitudeMismatch: u64 = 1006;
    /// @dev Error when bias sign vector does not match
    const EBiasesSignMismatch: u64 = 1007;
    /// @dev Error when weight magnitude and sign vector lengths do not match
    const EWeightsVectorLengthMismatch: u64 = 1008;
    /// @dev Error when bias magnitude and sign vector lengths do not match
    const EBiasesVectorLengthMismatch: u64 = 1009;
    /// @dev Error when scale value is 0
    const EInvalidScale: u64 = 1010;

    /// @notice Custom model creation function - Creates a model with user provided data
    /// @param graph Neural network graph
    /// @param layer_names List of layer names
    /// @param layer_in_dims List of input dimensions for each layer
    /// @param layer_out_dims List of output dimensions for each layer
    /// @param weights_magnitudes List of weight magnitudes for each layer
    /// @param weights_signs List of weight signs for each layer
    /// @param biases_magnitudes List of bias magnitudes for each layer
    /// @param biases_signs List of bias signs for each layer
    /// @param scale Fixed point scale (2^scale)
    public fun create_model(
        graph: &mut SignedFixedGraph, 
        layer_names: vector<vector<u8>>,
        layer_in_dims: vector<u64>,
        layer_out_dims: vector<u64>,
        weights_magnitudes: vector<vector<u64>>,
        weights_signs: vector<vector<u64>>,
        biases_magnitudes: vector<vector<u64>>,
        biases_signs: vector<vector<u64>>,
        scale: u64
    ) {
      // Validate scale value
      assert!(scale > 0, EInvalidScale);
      
      let layer_count = vector::length(&layer_names);
      
      // Check if all vectors have same length
      assert!(layer_count == vector::length(&layer_in_dims), EInputDimMismatch);
      assert!(layer_count == vector::length(&layer_out_dims), EOutputDimMismatch);
      assert!(layer_count == vector::length(&weights_magnitudes), EWeightsMagnitudeMismatch);
      assert!(layer_count == vector::length(&weights_signs), EWeightsSignMismatch);
      assert!(layer_count == vector::length(&biases_magnitudes), EBiasesMagnitudeMismatch);
      assert!(layer_count == vector::length(&biases_signs), EBiasesSignMismatch);
      
      let mut i = 0;
      while (i < layer_count) {
        let layer_name = *vector::borrow(&layer_names, i);
        let in_dim = *vector::borrow(&layer_in_dims, i);
        let out_dim = *vector::borrow(&layer_out_dims, i);
        
        // Validate weights and bias vector lengths
        let weights_mag = vector::borrow(&weights_magnitudes, i);
        let weights_sign = vector::borrow(&weights_signs, i);
        let biases_mag = vector::borrow(&biases_magnitudes, i);
        let biases_sign = vector::borrow(&biases_signs, i);
        
        assert!(vector::length(weights_mag) == vector::length(weights_sign), EWeightsVectorLengthMismatch);
        assert!(vector::length(biases_mag) == vector::length(biases_sign), EBiasesVectorLengthMismatch);
        assert!(vector::length(weights_mag) == in_dim * out_dim, EWeightsVectorLengthMismatch);
        assert!(vector::length(biases_mag) == out_dim, EBiasesVectorLengthMismatch);
        
        // Create layer
        graph::DenseSignedFixed(graph, in_dim, out_dim, layer_name, scale);
        
        // Set weights and biases
        graph::set_layer_weights_signed_fixed(
            graph,
            layer_name,
            *weights_mag,
            *weights_sign,
            *biases_mag,
            *biases_sign,
            in_dim, out_dim,
            scale
        );
        
        i = i + 1;
      };
    }

    /// @notice Custom model initialization function - Initializes a model with user provided data
    /// @param layer_names List of layer names
    /// @param layer_in_dims List of input dimensions for each layer
    /// @param layer_out_dims List of output dimensions for each layer
    /// @param weights_magnitudes List of weight magnitudes for each layer
    /// @param weights_signs List of weight signs for each layer
    /// @param biases_magnitudes List of bias magnitudes for each layer
    /// @param biases_signs List of bias signs for each layer
    /// @param scale Fixed point scale (2^scale)
    /// @param ctx Transaction context
    entry public fun initialize_model(
        layer_names: vector<vector<u8>>,
        layer_in_dims: vector<u64>,
        layer_out_dims: vector<u64>,
        weights_magnitudes: vector<vector<u64>>,
        weights_signs: vector<vector<u64>>,
        biases_magnitudes: vector<vector<u64>>,
        biases_signs: vector<vector<u64>>,
        scale: u64,
        ctx: &mut TxContext,
    ) {
      let mut graph = graph::create_signed_graph(ctx);
      
      create_model(
          &mut graph, 
          layer_names,
          layer_in_dims,
          layer_out_dims,
          weights_magnitudes,
          weights_signs,
          biases_magnitudes,
          biases_signs,
          scale
      );
      
      let mut partials = graph::create_partial_denses(ctx);
      graph::add_partials_for_all_but_last(&graph, &mut partials);
      transfer::public_share_object(graph);
      transfer::public_share_object(partials);
    }
}