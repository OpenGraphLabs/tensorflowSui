module models::model {
    use sui::tx_context::TxContext;
    use tensorflowsui::graph;
    use tensorflowsui::tensor;

    public fun create_model_signed_fixed(graph: &mut graph::SignedFixedGraph, scale: u64) {
        graph::DenseSignedFixed(graph, 49, 16, b"dense", scale);
        graph::DenseSignedFixed(graph, 16, 8, b"dense_1", scale);
        graph::DenseSignedFixed(graph, 8, 10, b"dense_2", scale);


        let wdense_mag = vector[23, 19, 8, 119, 5, 113, 23, 43, 41, 21, 5, 108, 9, 82, 16, 9, 25, 144, 162, 475, 212, 343, 50, 38, 304, 143, 490, 389, 499, 346, 71, 102, 6, 117, 10, 803, 41, 54, 145, 207, 150, 522, 589, 177, 256, 95, 395, 190, 644, 88, 105, 721, 21, 74, 333, 141, 87, 13, 143, 21, 207, 58, 90, 166, 496, 44, 160, 565, 83, 219, 143, 98, 121, 133, 109, 52, 246, 111, 103, 375, 522, 54, 172, 867, 353, 113, 46, 128, 327, 817, 197, 42, 126, 191, 526, 206, 26, 42, 152, 146, 4, 266, 266, 21, 89, 2, 340, 375, 85, 293, 260, 238, 201, 365, 335, 69, 357, 47, 66, 61, 32, 86, 161, 280, 156, 51, 316, 86, 26, 17, 4, 15, 120, 29, 21, 76, 39, 8, 63, 35, 45, 22, 36, 119, 3, 19, 6, 5, 76, 15, 5, 58, 1, 3, 129, 65, 4, 4, 8, 176, 0, 21, 21, 35, 35, 24, 4, 7, 0, 58, 22, 34, 99, 11, 4, 490, 38, 11, 85, 82, 81, 82, 11, 14, 30, 52, 15, 27, 89, 68, 8, 85, 8, 31, 92, 94, 23, 94, 136, 1, 87, 6, 39, 128, 99, 41, 33, 26, 171, 73, 44, 87, 249, 237, 611, 17, 62, 120, 85, 31, 623, 406, 56, 24, 5, 71, 47, 251, 92, 183, 135, 161, 48, 368, 195, 155, 141, 252, 224, 11, 148, 5, 20, 136, 232, 9, 31, 75, 92, 19, 145, 25, 11, 0, 24, 66, 97, 19, 2, 94, 164, 3, 117, 30, 84, 34, 82, 14, 64, 33, 2, 18, 7, 12, 29, 2, 167, 78, 12, 8, 41, 14, 13, 18, 63, 29, 5, 144, 43, 10, 26, 66, 10, 226, 42, 11, 6, 24, 58, 4, 24, 2, 3, 46, 0, 8, 14, 70, 9, 440, 79, 39, 7, 47, 38, 63, 54, 5, 12, 21, 619, 16, 94, 54, 17, 299, 363, 154, 222, 5, 50, 91, 323, 495, 116, 154, 73, 16, 67, 177, 373, 1, 123, 147, 118, 82, 134, 130, 199, 103, 246, 23, 214, 105, 43, 32, 10, 5, 99, 31, 38, 59, 2, 15, 20, 42, 2, 2, 114, 53, 8, 23, 11, 7, 88, 14, 62, 47, 30, 2, 45, 28, 1, 3, 47, 54, 13, 16, 25, 17, 86, 43, 28, 158, 1, 36, 36, 29, 36, 17, 20, 21, 16, 30, 13, 53, 50, 55, 63, 23, 35, 9, 13, 55, 4, 3, 27, 23, 40, 10, 38, 16, 3, 565, 70, 40, 11, 65, 45, 74, 34, 16, 48, 235, 267, 42, 27, 32, 140, 165, 93, 61, 33, 174, 28, 203, 1, 28, 185, 246, 262, 117, 80, 485, 220, 45, 84, 129, 0, 43, 287, 104, 94, 173, 46, 357, 27, 75, 51, 13, 73, 10, 130, 36, 0, 65, 70, 104, 9, 6, 14, 235, 6, 23, 19, 6, 79, 24, 14, 60, 28, 29, 3, 13, 12, 25, 21, 87, 91, 41, 32, 3, 5, 26, 39, 128, 6, 25, 20, 6, 57, 5, 22, 30, 11, 9, 7, 15, 15, 38, 114, 31, 14, 14, 88, 15, 74, 2, 8, 9, 48, 58, 56, 21, 34, 732, 110, 4, 56, 34, 55, 46, 58, 58, 38, 195, 882, 63, 114, 29, 113, 57, 176, 58, 88, 315, 182, 88, 90, 12, 142, 128, 140, 71, 161, 133, 214, 39, 210, 178, 335, 29, 55, 156, 76, 33, 53, 156, 52, 15, 48, 0, 12, 8, 41, 30, 243, 65, 44, 73, 59, 78, 9, 70, 48, 10, 19, 1, 33, 3, 42, 52, 83, 60, 50, 28, 31, 25, 1, 12, 113, 43, 27, 17, 8, 14, 71, 67, 26, 10, 67, 54, 62, 11, 9, 14, 41, 12, 18, 14, 2, 14, 88, 73, 63, 59, 105, 0, 265, 17, 64, 35, 41, 26, 80, 14, 98, 547, 100, 107, 44, 68, 235, 1, 72, 15, 130, 106, 573, 102, 188, 15, 85, 197, 216, 13, 101, 152, 84, 112, 50, 15, 19, 333, 282, 291, 56, 39, 131, 231, 437, 223, 271, 370, 339, 6, 121, 263, 96, 139, 276, 95, 4, 116, 253, 83, 106, 151, 49, 139, 92, 21, 227, 286, 1, 69, 166, 177, 85, 86, 22, 8, 59, 179, 95, 52, 96, 110, 239, 28, 10, 36, 100, 214, 87, 54, 12, 13, 17, 10, 98, 125, 4, 80, 100, 11, 56, 6, 320, 141, 47, 53, 64, 237, 16, 9, 132, 662, 43, 32, 429, 16, 174, 176, 183, 133, 24, 109, 82, 115, 229, 37, 51, 414, 188, 139, 54, 65, 38, 82, 173, 15, 58, 15, 26, 148, 242, 84, 181, 74, 46, 134, 21, 68];
        let wdense_sign = vector[1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0];
        let bdense_mag = vector[36, 8, 22, 59, 1, 1, 91, 41, 13, 11, 89, 65, 43, 81, 10, 10];
        let bdense_sign = vector[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        graph::set_layer_weights_signed_fixed(
            graph,
            b"dense",
            wdense_mag, wdense_sign,
            bdense_mag, bdense_sign,
            49, 16,
            scale
        );

        let wdense_1_mag = vector[111, 106, 134, 40, 18, 77, 4, 65, 34, 126, 46, 35, 134, 32, 12, 178, 47, 71, 45, 43, 43, 147, 144, 39, 15, 113, 39, 72, 67, 59, 137, 125, 80, 3, 110, 143, 28, 134, 73, 10, 231, 152, 54, 61, 18, 14, 30, 16, 130, 56, 9, 48, 52, 9, 80, 124, 110, 169, 86, 16, 150, 61, 61, 3, 38, 86, 80, 89, 30, 63, 71, 113, 28, 3, 168, 16, 114, 103, 102, 104, 65, 59, 60, 169, 40, 68, 80, 56, 4, 62, 62, 28, 116, 63, 49, 112, 3, 16, 113, 17, 99, 100, 95, 15, 41, 159, 80, 19, 39, 13, 86, 35, 25, 104, 46, 14, 183, 72, 56, 161, 79, 123, 113, 258, 59, 122, 161, 97];
        let wdense_1_sign = vector[1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1];
        let bdense_1_mag = vector[57, 15, 27, 23, 56, 68, 19, 61];
        let bdense_1_sign = vector[1, 1, 0, 0, 0, 0, 1, 0];

        graph::set_layer_weights_signed_fixed(
            graph,
            b"dense_1",
            wdense_1_mag, wdense_1_sign,
            bdense_1_mag, bdense_1_sign,
            16, 8,
            scale
        );

        let wdense_2_mag = vector[94, 99, 90, 139, 57, 111, 131, 34, 44, 68, 98, 45, 52, 66, 70, 141, 16, 71, 2, 31, 103, 162, 153, 43, 52, 24, 113, 154, 54, 80, 104, 48, 38, 3, 197, 40, 195, 88, 5, 46, 16, 82, 161, 52, 12, 44, 74, 73, 8, 43, 134, 177, 38, 31, 75, 45, 67, 44, 63, 41, 22, 27, 30, 87, 88, 45, 113, 145, 39, 66, 168, 208, 22, 24, 109, 11, 212, 81, 100, 89];
        let wdense_2_sign = vector[0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0];
        let bdense_2_mag = vector[118, 304, 87, 260, 77, 19, 82, 17, 375, 151];
        let bdense_2_sign = vector[1, 0, 0, 0, 0, 1, 0, 0, 1, 1];

        graph::set_layer_weights_signed_fixed(
            graph,
            b"dense_2",
            wdense_2_mag, wdense_2_sign,
            bdense_2_mag, bdense_2_sign,
            8, 10,
            scale
        );

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
        create_model_signed_fixed(&mut graph, 2); 
        let mut partials = graph::create_partial_denses(ctx);
        graph::add_partials_for_all_but_last(&graph, &mut partials);
        graph::share_graph(graph);
        graph::share_partial(partials);
    }
}