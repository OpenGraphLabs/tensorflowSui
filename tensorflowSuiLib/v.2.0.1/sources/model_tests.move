#[test_only]
module tensorflowsui::model_tests {
    use sui::test_scenario::{Self as ts};
    use sui::test_utils::{assert_eq};
    use std::debug;
    use tensorflowsui::graph;
    use tensorflowsui::model;
    
    #[test]
    fun test_create_model() {
        let addr = @0x1;
        let mut scenario = ts::begin(addr);
        let ctx = ts::ctx(&mut scenario);
        
        // Prepare test data for model creation
        let layer_dimensions = vector[vector[4, 2], vector[2, 1]];
        
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
            b"Test Model",
            b"A test model for prediction",
            b"classification",
            layer_dimensions,
            weights_magnitudes,
            weights_signs,
            biases_magnitudes,
            biases_signs,
            2,
            ctx,
        );
        
        // Move to next transaction and verify model ownership
        ts::next_tx(&mut scenario, @0x1);
        {
            // Check if sender owns the model object
            assert!(ts::has_most_recent_for_sender<model::Model>(&scenario), 0);
            
            // Get the model object
            let model_obj = ts::take_from_sender<model::Model>(&scenario);
            
            // Verify model properties
            assert_eq(model::get_scale(&model_obj), 2);
            
            // Return the model
            ts::return_to_sender(&scenario, model_obj);
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
            
            // Input: [100, 200, 300] (all positive)
            let input_magnitude = vector[100, 200, 300];
            let input_sign = vector[0, 0, 0];
            
            // Run prediction
            let (result_mag, result_sign, class_idx) = model::predict(&model, input_magnitude, input_sign);
            
            // Expected calculation:
            // Layer 1: [100, 200, 300] * [[1, 2], [3, 4], [5, 6]] + [1, 1] = [2200, 2800] + [1, 1] => scaling [22, 28] + [1, 1] => [23, 29]
            // Apply ReLU: [23, 29]
            // Layer 2: [23, 29] * [[7], [8]] + [1] = [23*7 + 29*8] + [1] = [161 + 232] + [1] = [393] + [1] => scaling [3] + [1] = [4]
            // Expected output: [4], all positive
            
            // Check result length
            assert_eq(vector::length(&result_mag), 1);
            assert_eq(vector::length(&result_sign), 1);
            
            // Check result values (allowing for some rounding error due to fixed point math)
            let result_value = *vector::borrow(&result_mag, 0);
            debug::print(&result_value);
            debug::print_stack_trace();
            assert!(result_value == 4, 101);
            assert_eq(*vector::borrow(&result_sign, 0), 0); // Positive
            
            // Since there's only one output node, argmax should be 0
            assert_eq(class_idx, 0);
            
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
            
            // Input: [100, 200, 300] (all positive)
            let input_magnitude = vector[100, 200, 300];
            let input_sign = vector[0, 0, 0];
            
            // Calculate first layer
            let (layer1_mag, layer1_sign, layer1_argmax) = model::predict_layer(&model, 0, input_magnitude, input_sign);
            
            // Expected calculation for layer 1:
            // [100, 200, 300] * [[1, 2], [3, 4], [5, 6]] + [1, 1] = [2200, 2800] + [1, 1] => scaling [22, 28] + [1, 1] => [23, 29]
            // Apply ReLU: [23, 29]
            
            // Check first layer result
            assert_eq(vector::length(&layer1_mag), 2);
            assert_eq(vector::length(&layer1_sign), 2);
            assert_eq(option::is_none(&layer1_argmax), true); // First layer should not return argmax

            assert!(*vector::borrow(&layer1_mag, 0) == 23, 111);
            assert!(*vector::borrow(&layer1_mag, 1) == 29, 112);
            
            // Both values should be positive
            assert_eq(*vector::borrow(&layer1_sign, 0), 0);
            assert_eq(*vector::borrow(&layer1_sign, 1), 0);
            
            // Now calculate second layer with output from first layer
            let (layer2_mag, layer2_sign, layer2_argmax) = model::predict_layer(&model, 1, layer1_mag, layer1_sign);
            
            // Expected calculation for layer 2:
            // [23, 29] * [[7], [8]] + [1] = [23*7 + 29*8] + [1] = [161 + 232] + [1] = [393] + [1] => scaling [3] + [1] = [4]
            // Expected output: [4], all positive

            // Check second layer result
            assert_eq(vector::length(&layer2_mag), 1);
            assert_eq(vector::length(&layer2_sign), 1);
            
            // Verify calculation (allowing for rounding errors)
            assert!(*vector::borrow(&layer2_mag, 0) == 4, 113);
            assert_eq(*vector::borrow(&layer2_sign, 0), 0); // Positive

            // Since this is the last layer, argmax should be returned
            assert_eq(option::is_some(&layer2_argmax), true);
            assert_eq(*option::borrow(&layer2_argmax), 0);
            
            ts::return_to_sender(&scenario, model);
        };
        
        ts::end(scenario);
    }
    
    #[test]
    fun test_predict_layer_partial() {
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
        
        // Test the dimension-level prediction for first layer
        ts::next_tx(&mut scenario, @0x1);
        {
            let model = ts::take_from_sender<model::Model>(&scenario);
            
            // Input: [100, 200, 300] (all positive)
            let input_magnitude = vector[100, 200, 300];
            let input_sign = vector[0, 0, 0];
            
            // --------- first layer ---------
            let (result1_mag, result1_sign, dim_idx1, is_last_dim1) = model::predict_layer_partial(
                &model, 
                0,  // Layer 0 
                0,  // First output dimension
                input_magnitude, 
                input_sign
            );
            
            // Expected calculation for first dimension of layer 1:
            // Layer 1: [100, 200, 300] * [[1, 2], [3, 4], [5, 6]] + [1, 1] = [2200, 2800] + [1, 1] => scaling [22, 28] + [1, 1] => [23, 29]
            // No need to apply ReLU since it's positive
            
            // Expected calculation for first dimension of layer 1: 23
            assert_eq(dim_idx1, 0);
            assert_eq(is_last_dim1, false); // Not the last dimension
            assert!(result1_mag == 23, 121); // ~23 with some rounding allowance
            assert_eq(result1_sign, 0); // Positive
            
            // Calculate second output dimension of first layer
            let (result2_mag, result2_sign, dim_idx2, is_last_dim2) = model::predict_layer_partial(
                &model, 
                0,  // Layer 0
                1,  // Second output dimension
                input_magnitude, 
                input_sign
            );
            
            // Expected calculation for second dimension of layer 1: 29
            assert_eq(dim_idx2, 1);
            assert_eq(is_last_dim2, true); // Last dimension of first layer
            assert!(result2_mag == 29, 122); // ~29 with some rounding allowance
            assert_eq(result2_sign, 0); // Positive
            
            // --------- second layer ---------
            // Calculate the only output dimension of second layer
            let (result3_mag, result3_sign, dim_idx3, is_last_dim3) = model::predict_layer_partial(
                &model, 
                1,  // Layer 1
                0,  // First (and only) output dimension
                vector[result1_mag, result2_mag],  // Output from first layer
                vector[result1_sign, result2_sign]
            );
            
            // Expected calculation for second layer:
            // Layer 2: [23, 29] * [[7], [8]] + [1] = [23*7 + 29*8] + [1] = [161 + 232] + [1] = [393] + [1] => scaling [3] + [1] = [4]
            // Expected output: [4], all positive
            
            // Check result
            assert_eq(dim_idx3, 0);
            assert_eq(is_last_dim3, true); // Last dimension of last layer
            assert!(result3_mag == 4, 123);
            assert_eq(result3_sign, 0); // Positive
            
            // Compare with full-layer prediction
            let (layer1_mag, layer1_sign, _) = model::predict_layer(&model, 0, input_magnitude, input_sign);
            
            // Ensure the dimension-level predictions match the full layer prediction
            assert_eq(vector::length(&layer1_mag), 2);
            assert!((*vector::borrow(&layer1_mag, 0) == result1_mag) && (*vector::borrow(&layer1_mag, 1) == result2_mag), 124);
            
            ts::return_to_sender(&scenario, model);
        };
        
        ts::end(scenario);
    }
}
