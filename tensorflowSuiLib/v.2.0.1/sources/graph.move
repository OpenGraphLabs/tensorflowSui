module tensorflowsui::graph {
    use std::debug;
    
    const NONE : u64= 0;
    const RELU : u64= 1;
    const SOFTMAX : u64 = 2;

    use tensorflowsui::tensor::{
        SignedFixedTensor, get_scale,get_magnitude,get_shape,get_sign,
        create_signed_fixed_tensor,
        scale_up,
    };

    public struct Result_arg has copy, drop {
        value : u64
    }

    public struct SignedFixedLayer has copy, drop, store {
        layer_type: vector<u8>,
        in_dimension: u64,
        out_dimension: u64,
        weight_tensor: SignedFixedTensor,  
        bias_tensor: SignedFixedTensor,    
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

    public fun get_weight_tensor(layer: &SignedFixedLayer): &SignedFixedTensor {
        &layer.weight_tensor
    }

    public fun get_bias_tensor(layer: &SignedFixedLayer): &SignedFixedTensor {
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
        let weight_tensor = create_signed_fixed_tensor(
            vector[in_dimension, out_dimension],
            weight_magnitudes,
            weight_signs,
            scale
        );

        // Create bias tensor with user-provided values
        let bias_tensor = create_signed_fixed_tensor(
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

    public fun apply_dense_signed_fixed(
        input_tensor: &SignedFixedTensor,
        weight_tensor: &SignedFixedTensor,
        bias_tensor: &SignedFixedTensor
    ): SignedFixedTensor {

        let batch = *vector::borrow(&get_shape(input_tensor), 0);
        let in_dim = *vector::borrow(&get_shape(input_tensor), 1);
        let w_in = *vector::borrow(&get_shape(weight_tensor), 0);
        let w_out= *vector::borrow(&get_shape(weight_tensor), 1);
        let b_out= *vector::borrow(&get_shape(bias_tensor), 0);

        assert!(in_dim == w_in, 10001);
        assert!(w_out == b_out, 10002);

        let s =  get_scale(input_tensor);
        assert!(s == get_scale(weight_tensor), 10003);
        assert!(s == get_scale(bias_tensor),   10004);

        let mut out_shape = vector::empty<u64>();
        vector::push_back(&mut out_shape, batch);
        vector::push_back(&mut out_shape, w_out);

        let mut out_mag = vector::empty<u64>();
        let mut out_sign= vector::empty<u64>();

        let mut b_idx = 0;
        while (b_idx < batch) {
            let mut j_idx = 0;
            while (j_idx < w_out) {

                let mut acc_sgn = 0;
                let mut acc_mag = 0;
                let mut i_idx = 0;
                while (i_idx < in_dim) {
                    let in_index = b_idx*in_dim + i_idx;
                    let w_index  = i_idx*w_out + j_idx;
                                                   
                    let in_s = *vector::borrow(& get_sign(input_tensor), in_index);
                    let in_m = *vector::borrow(&get_magnitude(input_tensor), in_index);
                    let w_s  = *vector::borrow(&get_sign(weight_tensor), w_index);
                    let w_m  = *vector::borrow(&get_magnitude(weight_tensor), w_index);

                    let mul_s = if (in_s == w_s) { 0 } else { 1 };
                    let mul_m = in_m * w_m;

                    let (acc2_s, acc2_m) = signed_add_element(
                        acc_sgn, acc_mag,
                        mul_s,   mul_m
                    );
                    acc_sgn = acc2_s;
                    acc_mag = acc2_m;

                    i_idx = i_idx + 1;
                };

                let factor = scale_up(1, s);
                let b_s  = *vector::borrow(&get_sign(bias_tensor), j_idx);
                let b_m  = *vector::borrow(&get_magnitude(bias_tensor), j_idx);
                let b_m_2s = b_m * factor;

                let (acc3_s, acc3_m) = signed_add_element(
                    acc_sgn, acc_mag,
                    b_s,     b_m_2s
                );
                let mut final_s = acc3_s;
                let mut final_m = acc3_m;
                if (final_s == 1) {
                    final_s = 0;
                    final_m = 0;
                };


                let divisor = scale_up(1, s);
                let rounded_m = final_m / divisor;

                vector::push_back(&mut out_sign, final_s);
                vector::push_back(&mut out_mag,  rounded_m);

                j_idx = j_idx + 1;
            };
            b_idx = b_idx + 1;
        };

        create_signed_fixed_tensor(out_shape, out_mag, out_sign, s)
    }

fun apply_relu_element(sign: u64, mag: u64): (u64, u64) {
    if (sign == 1) {
        (0, 0)
    } else {
        (sign, mag)
    }
}


public fun apply_dense_signed_fixed_2(
    input_tensor: &SignedFixedTensor,
    weight_tensor: &SignedFixedTensor,
    bias_tensor:   &SignedFixedTensor,
    activation_type: u64  
): SignedFixedTensor {
    let batch = *vector::borrow(&get_shape(input_tensor), 0);
    let in_dim = *vector::borrow(&get_shape(input_tensor), 1);
    let w_in = *vector::borrow(&get_shape(weight_tensor), 0);
    let w_out= *vector::borrow(&get_shape(weight_tensor), 1);
    let b_out= *vector::borrow(&get_shape(bias_tensor), 0);

    assert!(in_dim == w_in, 10001);
    assert!(w_out == b_out, 10002);

    let s =  get_scale(input_tensor);
    assert!(s == get_scale(weight_tensor), 10003);
    assert!(s == get_scale(bias_tensor),   10004);

    let mut out_shape = vector::empty<u64>();
    vector::push_back(&mut out_shape, batch);
    vector::push_back(&mut out_shape, w_out);

    let mut out_mag = vector::empty<u64>();
    let mut out_sign= vector::empty<u64>();

    let mut b_idx = 0;
    while (b_idx < batch) {
        let mut j_idx = 0;
        while (j_idx < w_out) {

            let mut acc_sgn = 0;
            let mut acc_mag = 0;

            let mut i_idx = 0;
            while (i_idx < in_dim) {
                let in_index = b_idx*in_dim + i_idx;
                let w_index  = i_idx*w_out + j_idx;
                                                   
                let in_s = *vector::borrow(& get_sign(input_tensor), in_index);
                let in_m = *vector::borrow(&get_magnitude(input_tensor), in_index);
                let w_s  = *vector::borrow(&get_sign(weight_tensor), w_index);
                let w_m  = *vector::borrow(&get_magnitude(weight_tensor), w_index);

                // Multiplication (scale=2s)
                let mul_s = if (in_s == w_s) { 0 } else { 1 };
                let mul_m = in_m * w_m;

                let (acc2_s, acc2_m) = signed_add_element(
                    acc_sgn, acc_mag,
                    mul_s,   mul_m
                );
                acc_sgn = acc2_s;
                acc_mag = acc2_m;

                i_idx = i_idx + 1;
            };

            let factor = scale_up(1, s);
            let b_s  = *vector::borrow(&get_sign(bias_tensor), j_idx);
            let b_m  = *vector::borrow(&get_magnitude(bias_tensor), j_idx);
            let b_m_2s = b_m * factor;

            let (acc3_s, acc3_m) = signed_add_element(
                acc_sgn, acc_mag,
                b_s,     b_m_2s
            );

            let (final_s, final_m) = if (activation_type == RELU) {
                apply_relu_element(acc3_s, acc3_m)
            } else {
                (acc3_s, acc3_m)
            };


            let divisor = scale_up(1, s);
            let rounded_m = final_m / divisor;

            vector::push_back(&mut out_sign, final_s);
            vector::push_back(&mut out_mag,  rounded_m);

            j_idx = j_idx + 1;
        };
        b_idx = b_idx + 1;
    };

    create_signed_fixed_tensor(out_shape, out_mag, out_sign, s)
}

    fun signed_add_element(
        s1: u64, m1: u64,
        s2: u64, m2: u64
    ): (u64, u64) {
        if (s1 == s2) {
            (s1, m1 + m2)
        } else {
            if (m1 >= m2) {
                (s1, m1 - m2)
            } else {
                (s2, m2 - m1)
            }
        }
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

        std::debug::print(&std::string::utf8(b"input vector:"));
        debug::print(&inputs);

        std::debug::print(&std::string::utf8(b"input number:"));
        debug::print(&input_size);
        
        std::debug::print(&std::string::utf8(b"output number:"));
        debug::print(&output_nodes);

        std::debug::print(&std::string::utf8(b"max computation:"));
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
           
            std::debug::print(&std::string::utf8(b"i number:"));
            debug::print(&i);

            std::debug::print(&std::string::utf8(b"j number:"));
            debug::print(&j);

            std::debug::print(&std::string::utf8(b"weigth_index:"));
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

    public struct PartialDense has  copy, drop, store {
        accum_mag: vector<u64>,   
        accum_sign: vector<u64>,  
        out_dim: u64,
        in_dim: u64,
        scale: u64,
    }

    public struct PartialDenses has key, store {
        id: UID,
        partials: vector<PartialDense>,
    }


    public fun create_partial_denses(ctx: &mut TxContext): PartialDenses {
        PartialDenses {
            id: object::new(ctx),
            partials: vector::empty<PartialDense>()
        }
    }

    entry public fun add_partial_for_layer(
        graph_obj: &SignedFixedGraph,
        layer_idx: u64,
        partial_denses: &mut PartialDenses
    ) {
        let layer_ref = get_layer_at(graph_obj, layer_idx);
        let in_dim = get_layer_in_dim(layer_ref);
        let out_dim = get_layer_out_dim(layer_ref);
        let s = 2;

        let mut mag = vector::empty<u64>();
        let mut sgn = vector::empty<u64>();
        let mut i = 0;
        while (i < out_dim) {
            vector::push_back(&mut mag, 0);
            vector::push_back(&mut sgn, 0);
            i = i + 1;
        };

        let new_partial = PartialDense {
            accum_mag: mag,
            accum_sign: sgn,
            out_dim,
            in_dim,
            scale: s,
        };

        
        vector::push_back(&mut partial_denses.partials, new_partial);
    }

    public fun add_partials_for_all_but_last(
        graph_obj: &SignedFixedGraph,
        partial_denses: &mut PartialDenses
    ) {
        let total = get_layer_count(graph_obj);
        let mut i = 0;
        while (i < (total - 1)) {
            add_partial_for_layer(graph_obj, i, partial_denses);
            i = i + 1;
        }
    }

    public entry fun init_partials(graph: &SignedFixedGraph, ctx: &mut TxContext) {
        let mut partials = create_partial_denses(ctx);
        add_partials_for_all_but_last(graph, &mut partials);
        transfer::public_share_object(partials);
    }
}
