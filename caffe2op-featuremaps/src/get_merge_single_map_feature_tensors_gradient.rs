crate::ix!();

/**
  | Explode given multi-feature tensors
  | with map features into multiple single-feature
  | tensor.
  |
  */
pub struct GetMergeSingleMapFeatureTensorsGradient;

num_inputs!{MergeSingleMapFeatureTensorsGradient, 
    |n: i32| {
        n >= 3 && n % 2 == 1
    }
}

num_outputs!{MergeSingleMapFeatureTensorsGradient, 
    |n: i32| {
        n >= 1
    }
}

inputs!{MergeSingleMapFeatureTensorsGradient, 
    0 => ("in1_lengths",            ".lengths"),
    1 => ("in1_presence",           ".presence"),
    2 => ("out_values_values_grad", ".values.values_grad")
}

outputs!{MergeSingleMapFeatureTensorsGradient, 
    0 => ("in1_values_grad",        ".values_grad")
}
