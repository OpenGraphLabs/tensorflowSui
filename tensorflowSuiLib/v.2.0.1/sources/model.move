// Copyright (c) OpenGraph, Inc.
// SPDX-License-Identifier: Apache-2.0

/// @title Fully Onchain Neural Network Inference Implementation
module tensorflowsui::model {
    use tensorflowsui::graph::{Self, SignedFixedGraph, PartialDenses};
    use std::string::{Self, String};
    use tensorflowsui::tensor::{Self, create_signed_fixed_tensor};
    use sui::event;
    
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
    /// @dev Error when input vector length does not match first layer input dimension
    const EInputDimensionMismatch: u64 = 1012;
    /// @dev Error when model object is invalid
    const EInvalidModel: u64 = 1013;
    /// @dev Error when model has no graphs
    const EModelHasNoGraphs: u64 = 1014;
    /// @dev Error when layer index is out of bounds
    const ELayerIndexOutOfBounds: u64 = 1015;
    /// @dev Error when dimension index is out of bounds
    const EDimensionIndexOutOfBounds: u64 = 1016;


    public struct Model has key {
      id: UID,
      name: String,
      description: String,
      task_type: String,
      graphs: vector<SignedFixedGraph>,
      partial_denses: vector<PartialDenses>,
      scale: u64,
    }
    
    /// @notice Event emitted when a layer computation is completed
    public struct LayerComputed has copy, drop {
        model_id: address,
        layer_idx: u64,
        output_magnitude: vector<u64>,
        output_sign: vector<u64>,
        activation_type: u64,
    }
    
    /// @notice Event emitted when model prediction is complete
    public struct PredictionCompleted has copy, drop {
        model_id: address,
        output_magnitude: vector<u64>,
        output_sign: vector<u64>,
        argmax_idx: u64,
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

      // Convert vector<u8> to String
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
        
        // Create layer and add to graph with user-provided weights and biases
        graph::build_signed_fixed_layer(
            &mut model.graphs[0], 
            in_dimension, 
            out_dimension, 
            *weights_magnitude, 
            *weights_sign, 
            *biases_magnitude, 
            *biases_sign, 
            scale
        );
        
        layer_idx = layer_idx + 1;
    };
    
        let mut partials = graph::create_partial_denses(ctx);
        graph::add_partials_for_all_but_last(&model.graphs[0], &mut partials);
        vector::push_back(&mut model.partial_denses, partials);

        transfer::transfer(model, tx_context::sender(ctx));
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

    /// @notice Run inference on the model with provided input
    /// @param model Model object to run inference on
    /// @param input_magnitude Magnitude values of the input vector
    /// @param input_sign Sign values of the input vector (0 for positive, 1 for negative)
    /// @return Tuple of (magnitude vector, sign vector, argmax index) of the model output
    entry public fun predict(
        model: &Model,
        input_magnitude: vector<u64>,
        input_sign: vector<u64>
    ): (vector<u64>, vector<u64>, u64) {
        // Validate model has at least one graph
        assert!(vector::length(&model.graphs) > 0, EModelHasNoGraphs);
        
        // Get the first graph (currently we only support one graph per model)
        let graph = vector::borrow(&model.graphs, 0);
        
        // Get first layer to validate input dimensions
        assert!(graph::get_layer_count(graph) > 0, EInvalidModel);
        let first_layer = graph::get_layer_at(graph, 0);
        let input_dim = graph::get_layer_in_dim(first_layer);
        
        // Validate input dimensions
        assert!(vector::length(&input_magnitude) == input_dim, EInputDimensionMismatch);
        assert!(vector::length(&input_sign) == input_dim, EInputDimensionMismatch);
        
        // Create input tensor (batch size 1)
        let input_shape = vector[1, input_dim];
        let input_tensor = create_signed_fixed_tensor(
            input_shape,
            input_magnitude,
            input_sign,
            model.scale
        );
        
        // Process through all layers in the graph
        let mut current_tensor = input_tensor;
        let layer_count = graph::get_layer_count(graph);
        
        let mut i = 0;
        while (i < layer_count) {
            let layer = graph::get_layer_at(graph, i);
            let weight_tensor = graph::get_weight_tensor(layer);
            let bias_tensor = graph::get_bias_tensor(layer);
            
            // Apply activation function (ReLU for all layers except the last one)
            let activation_type = if (i == layer_count - 1) { 0 } else { 1 }; // 0=None, 1=ReLU
            
            // Apply layer computation
            // TODO: select computation function based on layer type (dense, conv, etc.)
            current_tensor = graph::apply_dense_signed_fixed_2(
                &current_tensor,
                weight_tensor,
                bias_tensor,
                activation_type
            );
            
            i = i + 1;
        };
        
        // Extract results from the final tensor
        let result_mag = tensor::get_magnitude(&current_tensor);
        let result_sign = tensor::get_sign(&current_tensor);
        
        // Find argmax if we have results
        let max_idx = find_argmax(&result_mag, &result_sign);
        
        // Emit prediction completed event
        event::emit(PredictionCompleted {
            model_id: object::id_address(model),
            output_magnitude: result_mag,
            output_sign: result_sign,
            argmax_idx: max_idx,
        });
        
        (result_mag, result_sign, max_idx)
    }
    
    /// @notice Helper function to find the argmax index in result vectors
    fun find_argmax(magnitudes: &vector<u64>, signs: &vector<u64>): u64 {
        let mut max_idx = 0;
        let mut max_val = 0;
        let result_len = vector::length(magnitudes);
        
        if (result_len > 0) {
            let mut j = 0;
            while (j < result_len) {
                let val = vector::borrow(magnitudes, j);
                let sign = vector::borrow(signs, j);
                
                // Only consider positive values or zero for argmax
                if (*sign == 0 && *val > max_val) {
                    max_val = *val;
                    max_idx = j;
                };
                
                j = j + 1;
            };
        };
        
        max_idx
    }
    
    /// @notice Process a single layer and emit result as event (gas efficient version)
    /// @param model Model object to run inference on
    /// @param layer_idx Index of the layer to process
    /// @param input_magnitude Magnitude values of the input vector
    /// @param input_sign Sign values of the input vector
    /// @return Tuple of (magnitude vector, sign vector, optional argmax index for final layer)
    entry public fun predict_layer(
        model: &Model,
        layer_idx: u64,
        input_magnitude: vector<u64>,
        input_sign: vector<u64>
    ): (vector<u64>, vector<u64>, Option<u64>) {
        // Validate model has at least one graph
        assert!(vector::length(&model.graphs) > 0, EModelHasNoGraphs);
        
        // Get the first graph (currently we only support one graph per model)
        let graph = vector::borrow(&model.graphs, 0);
        
        // Check if layer_idx is valid
        let layer_count = graph::get_layer_count(graph);
        assert!(layer_idx < layer_count, ELayerIndexOutOfBounds);
        
        // Check if this is the last layer
        let is_last_layer = layer_idx == layer_count - 1;
        
        // Get the target layer
        let layer = graph::get_layer_at(graph, layer_idx);
        let input_dim = graph::get_layer_in_dim(layer);
        
        // Validate input dimensions
        assert!(vector::length(&input_magnitude) == input_dim, EInputDimensionMismatch);
        assert!(vector::length(&input_sign) == input_dim, EInputDimensionMismatch);
        
        // Create input tensor (batch size 1)
        let input_shape = vector[1, input_dim];
        let input_tensor = create_signed_fixed_tensor(
            input_shape,
            input_magnitude,
            input_sign,
            model.scale
        );
        
        // Get layer tensors
        let weight_tensor = graph::get_weight_tensor(layer);
        let bias_tensor = graph::get_bias_tensor(layer);
        
        // Apply activation function (ReLU for all layers except the last one)
        let activation_type = if (is_last_layer) { 0 } else { 1 }; // 0=None, 1=ReLU
        
        // Apply layer computation
        let result_tensor = graph::apply_dense_signed_fixed_2(
            &input_tensor,
            weight_tensor,
            bias_tensor,
            activation_type
        );
        
        // Extract results from the layer output tensor
        let result_mag = tensor::get_magnitude(&result_tensor);
        let result_sign = tensor::get_sign(&result_tensor);
        
        // For the last layer, calculate the argmax
        let mut argmax_idx = option::none();
        
        if (is_last_layer) {
            // Find argmax if we have results
            let max_idx = find_argmax(&result_mag, &result_sign);
            
            // Emit prediction completed event
            event::emit(PredictionCompleted {
                model_id: object::id_address(model),
                output_magnitude: result_mag,
                output_sign: result_sign,
                argmax_idx: max_idx,
            });
            
            argmax_idx = option::some(max_idx);
        };

        // Emit layer computed event
        event::emit(LayerComputed {
            model_id: object::id_address(model),
            layer_idx,
            output_magnitude: result_mag,
            output_sign: result_sign,
            activation_type,
        });
        
        (result_mag, result_sign, argmax_idx)
    }

    // TODO: implement partial dense layer computation
    /// @notice Process a single layer and emit result as event (gas efficient version)
    /// @param model Model object to run inference on
    /// @param layer_idx Index of the layer to process
    /// @param dimension_idx Index of the dimension to process
    /// @param input_magnitude Magnitude values of the input vector
    /// @param input_sign Sign values of the input vector
    /// @return Tuple of (magnitude vector, sign vector, optional argmax index for final layer)
    entry public fun predict_layer_partial(
        model: &Model,
        layer_idx: u64,
        dimension_idx: u64,
        input_magnitude: vector<u64>,
        input_sign: vector<u64>
    ): (vector<u64>, vector<u64>, Option<u64>) {
        // Validate model has at least one graph
        assert!(vector::length(&model.graphs) > 0, EModelHasNoGraphs);
        
        // Get the first graph (currently we only support one graph per model)
        let graph = vector::borrow(&model.graphs, 0);
        
        // Check if layer_idx is valid
        let layer_count = graph::get_layer_count(graph);
        assert!(layer_idx < layer_count, ELayerIndexOutOfBounds);
        
        // Check if this is the last layer
        let is_last_layer = layer_idx == layer_count - 1;
        
        // Get the target layer
        let layer = graph::get_layer_at(graph, layer_idx);
        let input_dim = graph::get_layer_in_dim(layer);

        assert!(dimension_idx < input_dim, EDimensionIndexOutOfBounds);
        
        // Validate input dimensions
        assert!(vector::length(&input_magnitude) == input_dim, EInputDimensionMismatch);
        assert!(vector::length(&input_sign) == input_dim, EInputDimensionMismatch);
        
        // Create input tensor (batch size 1)
        let input_shape = vector[1, input_dim];
        let input_tensor = create_signed_fixed_tensor(
            input_shape,
            input_magnitude,
            input_sign,
            model.scale
        );
        
        // Get layer tensors
        let weight_tensor = graph::get_weight_tensor(layer);
        let bias_tensor = graph::get_bias_tensor(layer);
        
        // Apply activation function (ReLU for all layers except the last one)
        let activation_type = if (is_last_layer) { 0 } else { 1 }; // 0=None, 1=ReLU
        
        // Apply layer computation
        let result_tensor = graph::apply_dense_signed_fixed_2(
            &input_tensor,
            weight_tensor,
            bias_tensor,
            activation_type
        );
        
        // Extract results from the layer output tensor
        let result_mag = tensor::get_magnitude(&result_tensor);
        let result_sign = tensor::get_sign(&result_tensor);
        
        // For the last layer, calculate the argmax
        let mut argmax_idx = option::none();
        
        if (is_last_layer) {
            // Find argmax if we have results
            let max_idx = find_argmax(&result_mag, &result_sign);
            
            // Emit prediction completed event
            event::emit(PredictionCompleted {
                model_id: object::id_address(model),
                output_magnitude: result_mag,
                output_sign: result_sign,
                argmax_idx: max_idx,
            });
            
            argmax_idx = option::some(max_idx);
        };

        // Emit layer computed event
        event::emit(LayerComputed {
            model_id: object::id_address(model),
            layer_idx,
            output_magnitude: result_mag,
            output_sign: result_sign,
            activation_type,
        });
        
        (result_mag, result_sign, argmax_idx)
    }
}