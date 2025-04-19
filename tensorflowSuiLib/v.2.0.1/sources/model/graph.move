module tensorflowsui::graph {
    use std::debug;
    use std::string;
    
    // const NONE : u64= 0;
    const RELU : u64= 1;
    // const SOFTMAX : u64 = 2;

    // use tensorflowsui::tensor::{
    //     SignedFixedTensor, get_scale,get_magnitude,get_shape,get_sign,
    //     scale_up,
    // };
    use tensorflowsui::tensor;

    // Define error constants at the top of the module
    const ERR_DIMENSION_MISMATCH: u64 = 10001;
    const ERR_BIAS_DIMENSION_MISMATCH: u64 = 10002;
    const ERR_SCALE_MISMATCH: u64 = 10003;
    const ERR_SCALE_MISMATCH_BIAS: u64 = 10004;
    
    public struct SignedFixedLayer has copy, drop, store {
        layer_type: vector<u8>,
        in_dimension: u64,
        out_dimension: u64,
        weight_tensor: tensor::SignedFixedTensor,  
        bias_tensor: tensor::SignedFixedTensor,    
    }

    public struct SignedFixedGraph has key, store {
        id : UID,
        layers: vector<SignedFixedLayer>,
    }

    public fun create_signed_graph(ctx: &mut TxContext): SignedFixedGraph {
        SignedFixedGraph { id: object::new(ctx), layers: vector::empty<SignedFixedLayer>() }
    }

    public fun get_layer_at(graph: &SignedFixedGraph, idx: u64): &SignedFixedLayer {
        vector::borrow(&graph.layers, idx)
    }

    public fun get_weight_tensor(layer: &SignedFixedLayer): &tensor::SignedFixedTensor {
        &layer.weight_tensor
    }

    public fun get_bias_tensor(layer: &SignedFixedLayer): &tensor::SignedFixedTensor {
        &layer.bias_tensor
    }

    public fun get_layer_in_dim(layer: &SignedFixedLayer): u64 {
        layer.in_dimension
    }

    public fun get_layer_out_dim(layer: &SignedFixedLayer): u64 {
        layer.out_dimension
    }

    public fun build_signed_fixed_layer(
        graph: &mut SignedFixedGraph,
        in_dimension: u64,
        out_dimension: u64,
        weight_magnitudes: vector<u64>,
        weight_signs: vector<u64>,
        bias_magnitudes: vector<u64>,
        bias_signs: vector<u64>,
        scale: u64
    ) {
        // Create weight tensor with user-provided values
        let weight_tensor = tensor::create_signed_fixed_tensor(
            vector[in_dimension, out_dimension],
            weight_magnitudes,
            weight_signs,
            scale
        );

        // Create bias tensor with user-provided values
        let bias_tensor = tensor::create_signed_fixed_tensor(
            vector[out_dimension],
            bias_magnitudes,
            bias_signs,
            scale
        );

        let layer = SignedFixedLayer {
            layer_type: b"dense_sf",
            in_dimension,
            out_dimension,
            weight_tensor,
            bias_tensor
        };

        vector::push_back(&mut graph.layers, layer);
    }

    /// @notice Performs dense layer computation with optional activation function
    /// @param input_tensor Input tensor (batch_size x input_dimension)
    /// @param weight_tensor Weight tensor (input_dimension x output_dimension)
    /// @param bias_tensor Bias tensor (output_dimension)
    /// @param activation_type Activation function to apply (0=None, 1=ReLU, 2=Softmax)
    /// @return Result tensor (batch_size x output_dimension)
    public fun compute_dense_layer(
        input_tensor: &tensor::SignedFixedTensor,
        weight_tensor: &tensor::SignedFixedTensor,
        bias_tensor: &tensor::SignedFixedTensor,
        activation_type: u64
    ): tensor::SignedFixedTensor {
        // 1. Extract tensor dimensions and validate
        let batch_size = *vector::borrow(&tensor::get_shape(input_tensor), 0);
        let input_dim = *vector::borrow(&tensor::get_shape(input_tensor), 1);
        let weight_input_dim = *vector::borrow(&tensor::get_shape(weight_tensor), 0);
        let output_dim = *vector::borrow(&tensor::get_shape(weight_tensor), 1);
        let bias_dim = *vector::borrow(&tensor::get_shape(bias_tensor), 0);

        // Validate dimensions match
        assert!(input_dim == weight_input_dim, ERR_DIMENSION_MISMATCH);
        assert!(output_dim == bias_dim, ERR_BIAS_DIMENSION_MISMATCH);

        // Validate scales match
        let scale = tensor::get_scale(input_tensor);
        assert!(scale == tensor::get_scale(weight_tensor), ERR_SCALE_MISMATCH);
        assert!(scale == tensor::get_scale(bias_tensor), ERR_SCALE_MISMATCH_BIAS);

        // 2. Prepare output tensor
        let output_shape = vector[batch_size, output_dim];

        let mut output_magnitude = vector::empty<u64>();
        let mut output_sign = vector::empty<u64>();

        // Pre-compute scale factor
        let scale_factor = compute_scale_factor(scale);

        // 3. Compute output for each batch item and output dimension
        let mut batch_idx = 0;
        while (batch_idx < batch_size) {
            // Process each output neuron (dimension)
            let mut output_idx = 0;
            while (output_idx < output_dim) {
                // Initialize accumulators for weighted sum
                let mut acc_sign = 0; // 0: positive, 1: negative
                let mut acc_magnitude = 0;
                
                // Compute weighted sum across input dimensions
                let mut input_idx = 0;
                while (input_idx < input_dim) {
                    // Calculate flat indices for accessing 1D tensor storage
                    let input_flat_idx = batch_idx * input_dim + input_idx;
                    let weight_flat_idx = input_idx * output_dim + output_idx;
                    
                    // Extract input and weight values
                    let input_sign = *vector::borrow(&tensor::get_sign(input_tensor), input_flat_idx);
                    let input_magnitude = *vector::borrow(&tensor::get_magnitude(input_tensor), input_flat_idx);
                    let weight_sign = *vector::borrow(&tensor::get_sign(weight_tensor), weight_flat_idx);
                    let weight_magnitude = *vector::borrow(&tensor::get_magnitude(weight_tensor), weight_flat_idx);

                    // Perform signed multiplication (XOR for sign)
                    let product_sign = if (input_sign == weight_sign) { 0 } else { 1 };
                    let product_magnitude = input_magnitude * weight_magnitude;

                    // Scale down the product to match the correct scale
                    // Product is currently at scale^2, so divide by scale_factor to bring back to scale
                    let scaled_product_magnitude = product_magnitude / scale_factor;

                    // Add product to accumulator
                    let (new_acc_sign, new_acc_magnitude) = signed_add_element(
                        acc_sign, acc_magnitude,
                        product_sign, scaled_product_magnitude
                    );
                    acc_sign = new_acc_sign;
                    acc_magnitude = new_acc_magnitude;

                    input_idx = input_idx + 1;
                };

                // Add bias (no need to scale bias, it's already at the correct scale)
                let bias_sign = *vector::borrow(&tensor::get_sign(bias_tensor), output_idx);
                let bias_magnitude = *vector::borrow(&tensor::get_magnitude(bias_tensor), output_idx);
                
                // Add bias to accumulated value
                let (final_sign, final_magnitude) = signed_add_element(
                    acc_sign, acc_magnitude,
                    bias_sign, bias_magnitude
                );

                // Apply activation function if specified
                let mut result_sign = final_sign;
                let mut result_magnitude = final_magnitude;
                
                if (activation_type == RELU && result_sign == 1) {
                    // For ReLU, zero out negative values
                    result_sign = 0;
                    result_magnitude = 0;
                };
                // TODO: Softmax activation would be implemented here if needed

                // Result is already at the correct scale, no need to scale down again
                vector::push_back(&mut output_sign, result_sign);
                vector::push_back(&mut output_magnitude, result_magnitude);

                output_idx = output_idx + 1;
            };
            batch_idx = batch_idx + 1;
        };

        // Create and return result tensor
        tensor::create_signed_fixed_tensor(output_shape, output_magnitude, output_sign, scale)
    }

    /// @notice Helper function to add two signed values
    /// @param sign1 Sign of first value (0: positive, 1: negative)
    /// @param magnitude1 Magnitude of first value
    /// @param sign2 Sign of second value (0: positive, 1: negative)
    /// @param magnitude2 Magnitude of second value
    /// @return Tuple of (result_sign, result_magnitude)
    fun signed_add_element(
        s1: u64, m1: u64,
        s2: u64, m2: u64
    ): (u64, u64) {
        if (s1 == s2) {
            // Same sign: add magnitudes
            (s1, m1 + m2)
        } else {
            // Different signs: subtract magnitudes
            if (m1 >= m2) {
                // First value has larger magnitude, keep its sign
                (s1, m1 - m2)
            } else {
                // Second value has larger magnitude, use its sign
                (s2, m2 - m1)
            }
        }
    }

    /// @notice Helper function to compute scale factor
    /// @param scale Scale value
    /// @return 10^scale value
    fun compute_scale_factor(scale: u64): u64 {
        let mut factor = 1;
        let mut i = 0;
        while (i < scale) {
            factor = factor * 10;
            i = i + 1;
        };
        factor
    }

    public struct Layer has copy, drop {
        name: vector<u8>,          
        layer_type: vector<u8>,    
        input_nodes : u64,         
        output_nodes : u64,        
        weights: vector<u64>,      
        bias: vector<u64>,         
    }

    public struct Graph has drop {
        layers: vector<Layer>,     
    }

    public fun get_output_nodes(layer : &Layer) : u64 {
        layer.output_nodes 
    }

    public fun get_weights(layer: &Layer): vector<u64> {
        layer.weights 
    }

    public fun get_bias(layer: &Layer): vector<u64> {
        layer.bias 
    }

        public fun get_layer_count(graph: &SignedFixedGraph): u64 {
        vector::length(&graph.layers)
    }

    public fun get_layer_type(layer: &Layer): &vector<u8> {
        &layer.layer_type 
    }

    public fun get_name(layer: &Layer): &vector<u8> {
        &layer.name 
    }

    public fun create(): Graph {
        Graph { layers: vector::empty<Layer>() } 
    }

    public fun add_layer(graph: &mut Graph, name: vector<u8>, layer_type: vector<u8>, input_nodes:u64, output_nodes:u64  ) {
        let weights : vector<u64> = initialize_weights(input_nodes, output_nodes);
        let bias : vector<u64> = initialize_bias(output_nodes);
        let layer = Layer { name, layer_type, input_nodes, output_nodes, weights, bias };
        vector::push_back(&mut graph.layers, layer);
    }

    public fun initialize_weights(input_nodes: u64, output_nodes:u64 ) : vector<u64> {
        let mut weights = vector::empty<u64>();
        let mut i = 0;
        while ( i < input_nodes * output_nodes) {
            vector::push_back(&mut weights, 1);
            i = i +1;
        };
        weights
    }

    public fun initialize_bias(output_nodes: u64): vector<u64> {
        let mut bias = vector::empty<u64>();

        let mut i = 0;
        while (i < output_nodes) {
            vector::push_back(&mut bias, 0);
            i = i + 1;
        };

        bias
    }

    public fun ReLu(weighted_sum : u64): u64 {
        if (weighted_sum > 0) {
            weighted_sum
        } else {
            0
        }
    }

    public fun Dense(graph: &mut Graph, input_nodes: u64, output_nodes: u64, name: vector<u8>): Layer {

        let weights = initialize_weights(input_nodes, output_nodes);
        let bias = initialize_bias(output_nodes);

        let layer = Layer {
            name,
            layer_type: b"dense",
            input_nodes,
            output_nodes,
            weights,
            bias,
        };

        vector::push_back(&mut graph.layers, layer);
        layer
    }

    public fun Input(graph: &mut Graph, name: vector<u8>): Layer {
        let layer = Layer {
            name,
            layer_type: b"input",
            input_nodes: 0,
            output_nodes: 0,
            weights: vector::empty<u64>(),
            bias: vector::empty<u64>(),
        };

        vector::push_back(&mut graph.layers, layer);
        layer
    }

    public fun set_layer_weights(graph: &mut Graph, name: vector<u8>, weights: vector<u64>, bias: vector<u64>) {
        let len = vector::length(&graph.layers);
        let mut i = 0;
        while (i < len) {
            let layer = vector::borrow_mut(&mut graph.layers, i);
            if (layer.name == name) {
                layer.weights = weights;
                layer.bias = bias;
                return
            };
            i = i + 1;
        };
        abort 1
    }

    public fun get_layer(graph: &Graph, name: vector<u8>): &Layer {
        let mut i = 0;
        while (i < vector::length(&graph.layers)) {
            let layer = vector::borrow(&graph.layers, i);
            if (layer.name == name) {
                return layer
            };
            i = i + 1;
        };
        abort 1
    }

    /* Decription  */
    public fun apply_dense(inputs: vector<u64>, weights: &vector<u64>, bias: &vector<u64>, output_nodes: u64): vector<u64> {
    let mut result = vector::empty<u64>();
    let input_size = vector::length(&inputs);
    let max_computation = input_size * output_nodes;

        debug::print(&string::utf8(b"input vector:"));
        debug::print(&inputs);

        debug::print(&string::utf8(b"input number:"));
        debug::print(&input_size);
        
        debug::print(&string::utf8(b"output number:"));
        debug::print(&output_nodes);

        debug::print(&string::utf8(b"max computation:"));
        debug::print(&max_computation);

        debug::print(weights);
        debug::print(bias);
        
        debug::print(&output_nodes);

    let mut i = 0;
    while (i < output_nodes) {
        let mut weighted_sum = 0;
        let mut j = 0;

        while (j < input_size) {
            let weight_index = i * (input_size) + j;
           
            debug::print(&string::utf8(b"i number:"));
            debug::print(&i);

            debug::print(&string::utf8(b"j number:"));
            debug::print(&j);

            debug::print(&string::utf8(b"weigth_index:"));
            debug::print(& weight_index);

            weighted_sum = weighted_sum + (inputs[j] * weights[weight_index]);
            j = j + 1;
        };

        weighted_sum = weighted_sum + *vector::borrow(bias, i);
        weighted_sum = ReLu(weighted_sum);
        vector::push_back(&mut result, weighted_sum);
        i = i + 1;
    };

    result
}

    public fun apply_conv2d(prev_output: vector<u64>, weights: &vector<u64>, bias: u64): vector<u64> {
        let mut result = vector::empty<u64>();
        let kernel_size = vector::length(weights);
        let prev_output_size = vector::length(&prev_output);

        let mut i = 0;
        while (i <= prev_output_size - kernel_size) {
            let mut conv_sum = 0;
            let mut j = 0;
            while (j < kernel_size) {
                conv_sum = conv_sum + (prev_output[i + j] * weights[j]);
                j = j + 1;
            };
            conv_sum = conv_sum + bias;
            vector::push_back(&mut result, conv_sum);
            i = i + 1;
        };
        result
    }
    
}
