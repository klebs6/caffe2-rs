crate::ix!();

num_inputs!{MergeSingleListFeatureTensorsGradient, 
    |n: i32| {
        n >= 3 && n % 2 == 1
    }
}

num_outputs!{MergeSingleListFeatureTensorsGradient, 
    |n: i32| {
        n >= 1
    }
}

inputs!{MergeSingleListFeatureTensorsGradient, 
    0 => ("in1_lengths",        ".lengths"),
    1 => ("in1_presence",       ".presence"),
    2 => ("out_values_values",  ".values.values_grad")
}

outputs!{MergeSingleListFeatureTensorsGradient, 
    0 => ("out1_values",        ".values_grad")
}
