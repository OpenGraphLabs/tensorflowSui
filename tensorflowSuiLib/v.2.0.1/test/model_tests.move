module tensorflowsui::model_tests {
    use sui::test_scenario::{Self as ts, Scenario};
    use sui::test_utils::{assert_eq};
    use tensorflowsui::model;
    use tensorflowsui::graph::{Self, SignedFixedGraph};
    use std::vector;
    use std::option;

    #[test]
    fun test_create_model_signed_fixed() {
        let mut scenario = ts::begin(@0x1);
        let ctx = ts::ctx(&mut scenario);
        
        // Create graph
        let mut graph = graph::create_signed_graph(ctx);
        
        // Create model
        model::create_model_signed_fixed(&mut graph, 2);
        
        // Check layer count
        assert_eq(graph::get_layer_count(&graph), 3);
        
        // Check first layer
        let layer0 = graph::get_layer_at(&graph, 0);
        assert_eq(graph::get_layer_name(layer0), b"dense1");
        assert_eq(graph::get_layer_in_dim(layer0), 784);
        assert_eq(graph::get_layer_out_dim(layer0), 16);
        
        // Check second layer
        let layer1 = graph::get_layer_at(&graph, 1);
        assert_eq(graph::get_layer_name(layer1), b"dense2");
        assert_eq(graph::get_layer_in_dim(layer1), 16);
        assert_eq(graph::get_layer_out_dim(layer1), 8);
        
        // Check third layer
        let layer2 = graph::get_layer_at(&graph, 2);
        assert_eq(graph::get_layer_name(layer2), b"output");
        assert_eq(graph::get_layer_in_dim(layer2), 8);
        assert_eq(graph::get_layer_out_dim(layer2), 10);
        
        graph::share_graph(graph);
        ts::end(scenario);
    }
    
    #[test]
    fun test_create_model() {
        let mut scenario = ts::begin(@0x1);
        let ctx = ts::ctx(&mut scenario);
        
        // Create graph
        let mut graph = graph::create_signed_graph(ctx);
        
        // Prepare test data
        let layer_names = vector[b"layer1", b"layer2"];
        let layer_in_dims = vector[4, 2];
        let layer_out_dims = vector[2, 1];
        
        // First layer weights and biases
        let w1_mag = vector[1, 2, 3, 4, 5, 6, 7, 8];
        let w1_sign = vector[0, 0, 0, 0, 1, 1, 1, 1];
        let b1_mag = vector[1, 2];
        let b1_sign = vector[0, 1];
        
        // Second layer weights and biases
        let w2_mag = vector[9, 10];
        let w2_sign = vector[0, 1];
        let b2_mag = vector[3];
        let b2_sign = vector[0];
        
        let weights_magnitudes = vector[w1_mag, w2_mag];
        let weights_signs = vector[w1_sign, w2_sign];
        let biases_magnitudes = vector[b1_mag, b2_mag];
        let biases_signs = vector[b1_sign, b2_sign];
        
        // Create model
        model::create_model(
            &mut graph,
            layer_names,
            layer_in_dims,
            layer_out_dims,
            weights_magnitudes,
            weights_signs,
            biases_magnitudes,
            biases_signs,
            2
        );
        
        // Check layer count
        assert_eq(graph::get_layer_count(&graph), 2);
        
        // Check first layer
        let layer0 = graph::get_layer_at(&graph, 0);
        assert_eq(graph::get_layer_name(layer0), b"layer1");
        assert_eq(graph::get_layer_in_dim(layer0), 4);
        assert_eq(graph::get_layer_out_dim(layer0), 2);
        
        // Check second layer
        let layer1 = graph::get_layer_at(&graph, 1);
        assert_eq(graph::get_layer_name(layer1), b"layer2");
        assert_eq(graph::get_layer_in_dim(layer1), 2);
        assert_eq(graph::get_layer_out_dim(layer1), 1);
        
        transfer::public_share_object(graph);
        ts::end(scenario);
    }
    
    #[test]
    fun test_initialize() {
        let mut scenario = ts::begin(@0x1);
        
        // Call initialization function
        ts::next_tx(&mut scenario, @0x1);
        {
            let ctx = ts::ctx(&mut scenario);
            model::initialize(ctx);
        };
        
        // Check shared graph
        ts::next_tx(&mut scenario, @0x1);
        {
            assert!(ts::has_most_recent_shared<SignedFixedGraph>(), 0);
        };
        
        ts::end(scenario);
    }
    
    #[test]
    fun test_initialize_model() {
        let mut scenario = ts::begin(@0x1);
        
        // Prepare test data
        let layer_names = vector[b"layer1", b"layer2"];
        let layer_in_dims = vector[4, 2];
        let layer_out_dims = vector[2, 1];
        
        // First layer weights and biases
        let w1_mag = vector[1, 2, 3, 4, 5, 6, 7, 8];
        let w1_sign = vector[0, 0, 0, 0, 1, 1, 1, 1];
        let b1_mag = vector[1, 2];
        let b1_sign = vector[0, 1];
        
        // Second layer weights and biases
        let w2_mag = vector[9, 10];
        let w2_sign = vector[0, 1];
        let b2_mag = vector[3];
        let b2_sign = vector[0];
        
        let weights_magnitudes = vector[w1_mag, w2_mag];
        let weights_signs = vector[w1_sign, w2_sign];
        let biases_magnitudes = vector[b1_mag, b2_mag];
        let biases_signs = vector[b1_sign, b2_sign];
        
        // Call initialization function
        ts::next_tx(&mut scenario, @0x1);
        {
            let ctx = ts::ctx(&mut scenario);
            model::initialize_model(
                ctx,
                layer_names,
                layer_in_dims,
                layer_out_dims,
                weights_magnitudes,
                weights_signs,
                biases_magnitudes,
                biases_signs,
                2
            );
        };
        
        // Check shared graph
        ts::next_tx(&mut scenario, @0x1);
        {
            assert!(ts::has_most_recent_shared<SignedFixedGraph>(), 0);
        };
        
        ts::end(scenario);
    }

    #[test]
    fun test_predict() {
        let mut scenario = ts::begin(@0x1);
        
        // Create a test model first
        ts::next_tx(&mut scenario, @0x1);
        {
            let ctx = ts::ctx(&mut scenario);
            
            // Simple model with 2 layers: 3x2 and 2x1
            let name = b"Test Model";
            let description = b"A test model for prediction";
            let task_type = b"classification";
            
            // Layer dimensions: [[3, 2], [2, 1]]
            let mut layer_dimensions = vector::empty<vector<u64>>();
            vector::push_back(&mut layer_dimensions, vector[3, 2]);
            vector::push_back(&mut layer_dimensions, vector[2, 1]);
            
            // First layer weights (3x2): [[1, 2], [3, 4], [5, 6]]
            // Flattened: [1, 2, 3, 4, 5, 6]
            let mut weights_magnitudes = vector::empty<vector<u64>>();
            vector::push_back(&mut weights_magnitudes, vector[1, 2, 3, 4, 5, 6]);
            
            // All positive signs
            let mut weights_signs = vector::empty<vector<u64>>();
            vector::push_back(&mut weights_signs, vector[0, 0, 0, 0, 0, 0]);
            
            // Second layer weights (2x1): [[7], [8]]
            // Flattened: [7, 8]
            vector::push_back(&mut weights_magnitudes, vector[7, 8]);
            vector::push_back(&mut weights_signs, vector[0, 0]);
            
            // Biases
            let mut biases_magnitudes = vector::empty<vector<u64>>();
            vector::push_back(&mut biases_magnitudes, vector[1, 1]);  // First layer bias
            vector::push_back(&mut biases_magnitudes, vector[1]);     // Second layer bias
            
            // All positive bias signs
            let mut biases_signs = vector::empty<vector<u64>>();
            vector::push_back(&mut biases_signs, vector[0, 0]);
            vector::push_back(&mut biases_signs, vector[0]);
            
            model::create_model(
                name,
                description,
                task_type,
                layer_dimensions,
                weights_magnitudes,
                weights_signs,
                biases_magnitudes,
                biases_signs,
                2, // Scale factor
                ctx
            );
        };
        
        // Now test the prediction
        ts::next_tx(&mut scenario, @0x1);
        {
            let model = ts::take_from_sender<model::Model>(&scenario);
            
            // Input: [1, 2, 3] (all positive)
            let input_magnitude = vector[1, 2, 3];
            let input_sign = vector[0, 0, 0];
            
            // Run prediction
            let (result_mag, result_sign, class_idx) = model::predict(&model, input_magnitude, input_sign);
            
            // Expected calculation:
            // Layer 1: [1, 2, 3] * [[1, 2], [3, 4], [5, 6]] + [1, 1] = [22, 28]
            // Apply ReLU: [22, 28]
            // Layer 2: [22, 28] * [[7], [8]] + [1] = [22*7 + 28*8 + 1] = [155 + 224 + 1] = [380]
            // Expected output: [380], all positive
            
            // Check result length
            assert_eq(vector::length(&result_mag), 1);
            assert_eq(vector::length(&result_sign), 1);
            
            // Check result values (allowing for some rounding error due to fixed point math)
            let result_value = *vector::borrow(&result_mag, 0);
            assert!(result_value > 370 && result_value < 390, 101);
            assert_eq(*vector::borrow(&result_sign, 0), 0); // Positive
            
            // Since there's only one output node, argmax should be 0
            assert_eq(class_idx, 0);
            
            // Test predict_class
            let class_only = model::predict_class(&model, input_magnitude, input_sign);
            assert_eq(class_only, 0);
            
            ts::return_to_sender(&scenario, model);
        };
        
        ts::end(scenario);
    }
    
    #[test]
    fun test_predict_layer() {
        let mut scenario = ts::begin(@0x1);
        
        // Create a test model first
        ts::next_tx(&mut scenario, @0x1);
        {
            let ctx = ts::ctx(&mut scenario);
            
            // Simple model with 2 layers: 3x2 and 2x1
            let name = b"Test Model";
            let description = b"A test model for prediction";
            let task_type = b"classification";
            
            // Layer dimensions: [[3, 2], [2, 1]]
            let mut layer_dimensions = vector::empty<vector<u64>>();
            vector::push_back(&mut layer_dimensions, vector[3, 2]);
            vector::push_back(&mut layer_dimensions, vector[2, 1]);
            
            // First layer weights (3x2): [[1, 2], [3, 4], [5, 6]]
            // Flattened: [1, 2, 3, 4, 5, 6]
            let mut weights_magnitudes = vector::empty<vector<u64>>();
            vector::push_back(&mut weights_magnitudes, vector[1, 2, 3, 4, 5, 6]);
            
            // All positive signs
            let mut weights_signs = vector::empty<vector<u64>>();
            vector::push_back(&mut weights_signs, vector[0, 0, 0, 0, 0, 0]);
            
            // Second layer weights (2x1): [[7], [8]]
            // Flattened: [7, 8]
            vector::push_back(&mut weights_magnitudes, vector[7, 8]);
            vector::push_back(&mut weights_signs, vector[0, 0]);
            
            // Biases
            let mut biases_magnitudes = vector::empty<vector<u64>>();
            vector::push_back(&mut biases_magnitudes, vector[1, 1]);  // First layer bias
            vector::push_back(&mut biases_magnitudes, vector[1]);     // Second layer bias
            
            // All positive bias signs
            let mut biases_signs = vector::empty<vector<u64>>();
            vector::push_back(&mut biases_signs, vector[0, 0]);
            vector::push_back(&mut biases_signs, vector[0]);
            
            model::create_model(
                name,
                description,
                task_type,
                layer_dimensions,
                weights_magnitudes,
                weights_signs,
                biases_magnitudes,
                biases_signs,
                2, // Scale factor
                ctx
            );
        };
        
        // Test the layer-by-layer prediction
        ts::next_tx(&mut scenario, @0x1);
        {
            let model = ts::take_from_sender<model::Model>(&scenario);
            
            // Input: [1, 2, 3] (all positive)
            let input_magnitude = vector[1, 2, 3];
            let input_sign = vector[0, 0, 0];
            
            // Calculate first layer
            let (layer1_mag, layer1_sign, layer1_argmax) = model::predict_layer(&model, 0, input_magnitude, input_sign);
            
            // Expected calculation for layer 1:
            // [1, 2, 3] * [[1, 2], [3, 4], [5, 6]] + [1, 1] = [22, 28]
            // Apply ReLU: [22, 28]
            
            // Check first layer result
            assert_eq(vector::length(&layer1_mag), 2);
            assert_eq(vector::length(&layer1_sign), 2);
            assert_eq(option::is_none(&layer1_argmax), true); // First layer should not return argmax
            
            // Verify calculation, allowing for some rounding errors
            assert!((*vector::borrow(&layer1_mag, 0) >= 21 && *vector::borrow(&layer1_mag, 0) <= 23), 101);
            assert!((*vector::borrow(&layer1_mag, 1) >= 27 && *vector::borrow(&layer1_mag, 1) <= 29), 102);
            
            // Both values should be positive
            assert_eq(*vector::borrow(&layer1_sign, 0), 0);
            assert_eq(*vector::borrow(&layer1_sign, 1), 0);
            
            // Now calculate second layer with output from first layer
            let (layer2_mag, layer2_sign, layer2_argmax) = model::predict_layer(&model, 1, layer1_mag, layer1_sign);
            
            // Expected calculation for layer 2:
            // [22, 28] * [[7], [8]] + [1] = [22*7 + 28*8 + 1] = [155 + 224 + 1] = [380]

            // Check second layer result
            assert_eq(vector::length(&layer2_mag), 1);
            assert_eq(vector::length(&layer2_sign), 1);
            
            // Verify calculation (allowing for rounding errors)
            assert!((*vector::borrow(&layer2_mag, 0) >= 375 && *vector::borrow(&layer2_mag, 0) <= 385), 103);
            assert_eq(*vector::borrow(&layer2_sign, 0), 0); // Positive

            // Since this is the last layer, argmax should be returned
            assert_eq(option::is_some(&layer2_argmax), true);
            assert_eq(option::extract(&layer2_argmax), 0); // Only one output, so argmax should be 0
            
            ts::return_to_sender(&scenario, model);
        };
        
        ts::end(scenario);
    }
}
