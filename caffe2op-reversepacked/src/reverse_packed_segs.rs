crate::ix!();

/**
  | Reverse segments in a 3-D tensor (lengths,
  | segments, embeddings,), leaving paddings
  | unchanged.
  | 
  | This operator is used to reverse input
  | of a recurrent neural network to make
  | it a BRNN.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_DISPATCH_HELPER]
pub struct ReversePackedSegsOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{ReversePackedSegs, 2}

num_outputs!{ReversePackedSegs, 1}

inputs!{ReversePackedSegs, 
    0 => ("data",    "a 3-D (lengths, segments, embeddings,) tensor."),
    1 => ("lengths", "length of each segment.")
}

outputs!{ReversePackedSegs, 
    0 => ("reversed data", "a (lengths, segments, embeddings,) tensor with each segment reversed and paddings unchanged.")
}

register_cpu_operator!{ReversePackedSegs, ReversePackedSegsOp<CPUContext>}

input_tags!{
    ReversePackedSegsOp {
        Data,
        Lengths
    }
}
