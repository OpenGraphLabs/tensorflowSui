// Copyright (c) OpenGraph, Inc.
// SPDX-License-Identifier: Apache-2.0

/// @title Fully Onchain Neural Network Inference Implementation
module tensorflowsui::model {
    use tensorflowsui::graph::{Self, SignedFixedGraph, PartialDenses};
    use std::string::{Self, String};
    
    /// @dev Error when dimension pair does not match
    const EDimensionPairMismatch: u64 = 1002;
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
    /// @dev Error when layer dimensions vector is empty
    const ELayerDimensionsEmpty: u64 = 1011;

    public struct Model has key {
      id: UID,
      name: String,
      description: String,
      task_type: String,
      graphs: vector<SignedFixedGraph>,
      partial_denses: vector<PartialDenses>,
      scale: u64,
    }

    /// @notice Custom model initialization function - creates a model with user provided data
    /// @param name Model name
    /// @param description Model description
    /// @param task_type Model task type (e.g., "classification", "regression")
    /// @param layer_dimensions List of [input_dim, output_dim] pairs for each layer
    /// @param weights_magnitudes List of weight magnitudes for each layer
    /// @param weights_signs List of weight signs for each layer
    /// @param biases_magnitudes List of bias magnitudes for each layer
    /// @param biases_signs List of bias signs for each layer
    /// @param scale Fixed point scale (2^scale)
    /// @param ctx Transaction context
    entry public fun create_model(
        name: vector<u8>,
        description: vector<u8>,
        task_type: vector<u8>,
        layer_dimensions: vector<vector<u64>>,
        weights_magnitudes: vector<vector<u64>>,
        weights_signs: vector<vector<u64>>,
        biases_magnitudes: vector<vector<u64>>,
        biases_signs: vector<vector<u64>>,
        scale: u64,
        ctx: &mut TxContext,
    ) {
      // Validate scale value
      assert!(scale > 0, EInvalidScale);
      
      let layer_count = vector::length(&layer_dimensions);
      assert!(layer_count > 0, ELayerDimensionsEmpty);
      
      // Check if all vectors have same length
      assert!(layer_count == vector::length(&weights_magnitudes), EWeightsMagnitudeMismatch);
      assert!(layer_count == vector::length(&weights_signs), EWeightsSignMismatch);
      assert!(layer_count == vector::length(&biases_magnitudes), EBiasesMagnitudeMismatch);
      assert!(layer_count == vector::length(&biases_signs), EBiasesSignMismatch);

      // vector<u8>를 String으로 변환
      let name_string = string::utf8(name);
      let description_string = string::utf8(description);
      let task_type_string = string::utf8(task_type);

      let mut model = Model {
        id: object::new(ctx),
        name: name_string,
        description: description_string,
        task_type: task_type_string,
        graphs: vector::empty<SignedFixedGraph>(),
        partial_denses: vector::empty<PartialDenses>(),
        scale,
      };

      // NOTE(jarry): currently, we handle only one graph
      let graph = graph::create_signed_graph(ctx);
      vector::push_back(&mut model.graphs, graph);
      
      let mut layer_idx = 0;
      while (layer_idx < layer_count) {
        // Get layer dimensions
        let dimension_pair = vector::borrow(&layer_dimensions, layer_idx);
        assert!(vector::length(dimension_pair) == 2, EDimensionPairMismatch); // Make sure the dimension pair is [in_dim, out_dim]
        
        let in_dimension = *vector::borrow(dimension_pair, 0);
        let out_dimension = *vector::borrow(dimension_pair, 1);
        
        // Validate weights and bias vector lengths
        let weights_magnitude = vector::borrow(&weights_magnitudes, layer_idx);
        let weights_sign = vector::borrow(&weights_signs, layer_idx);
        let biases_magnitude = vector::borrow(&biases_magnitudes, layer_idx);
        let biases_sign = vector::borrow(&biases_signs, layer_idx);
        
        assert!(vector::length(weights_magnitude) == vector::length(weights_sign), EWeightsVectorLengthMismatch);
        assert!(vector::length(biases_magnitude) == vector::length(biases_sign), EBiasesVectorLengthMismatch);
        assert!(vector::length(weights_magnitude) == in_dimension * out_dimension, EWeightsVectorLengthMismatch);
        assert!(vector::length(biases_magnitude) == out_dimension, EBiasesVectorLengthMismatch);
        
        // Create layer and add to graph
        graph::build_signed_fixed_layer(&mut model.graphs[0], in_dimension, out_dimension, scale);
        
        layer_idx = layer_idx + 1;
      };
      
      let mut partials = graph::create_partial_denses(ctx);
      graph::add_partials_for_all_but_last(&model.graphs[0], &mut partials);
      vector::push_back(&mut model.partial_denses, partials);

      transfer::transfer(model, ctx.sender());
    }

    /// @notice Helper function to get model name as String
    /// @param model Model object
    /// @return Name of the model
    public fun get_name(model: &Model): &String {
        &model.name
    }

    /// @notice Helper function to get model description as String
    /// @param model Model object
    /// @return Description of the model
    public fun get_description(model: &Model): &String {
        &model.description
    }

    /// @notice Helper function to get model task type as String
    /// @param model Model object
    /// @return Task type of the model (e.g., "classification", "regression")
    public fun get_task_type(model: &Model): &String {
        &model.task_type
    }

    /// @notice Helper function to get model scale
    /// @param model Model object
    /// @return Scale value used for fixed-point calculations
    public fun get_scale(model: &Model): u64 {
        model.scale
    }
}