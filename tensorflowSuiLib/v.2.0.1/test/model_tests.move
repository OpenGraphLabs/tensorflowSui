module tensorflowsui::model_tests {
    use sui::test_scenario::{Self as ts, Scenario};
    use sui::test_utils::{assert_eq};
    use tensorflowsui::model;
    use tensorflowsui::graph::{Self, SignedFixedGraph};
    use std::vector;

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
}
