crate::ix!();

num_inputs!{MergeMultiMapFeatureTensorsGradient, 
    |n: i32| {
        n >= 3 && n % 2 == 1
    }
}

num_outputs!{MergeMultiMapFeatureTensorsGradient, 
    |n: i32| {
        n >= 1
    }
}

inputs!{MergeMultiMapFeatureTensorsGradient, 
    0 => ("in1_lengths",             ".lengths"),
    1 => ("in1_values_lengths",      ".values.lengths"),
    2 => ("out_values_values_grad",  ".values.values_grad")
}

outputs!{MergeMultiMapFeatureTensorsGradient, 
    0 => ("in1_values_values_grad",  ".values.values_grad")
}
