crate::ix!();

num_inputs!{UnpackSegments, 2}

num_outputs!{UnpackSegments, 1}

inputs!{UnpackSegments, 
    0 => ("lengths",       "1-d int/long tensor contains the length in each of the input."),
    1 => ("tensor",        "N+1 dim Tensor.")
}

outputs!{UnpackSegments, 
    0 => ("packed_tensor", "N dim Tensor")
}

args!{UnpackSegments, 
    0 => ("max_length",    "The pre-defined max_length for the packed segments")
}

input_tags!{
    UnpackSegmentsOp {
        Lengths,
        Data
    }
}
