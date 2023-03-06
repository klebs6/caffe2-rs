crate::ix!();

/**
  | Apply TopK to each segment of the input
  | tensor, where segments are defined
  | by their LENGTHS, and concatenate them
  | in an output tensor of shape=(SIZE(LENGTHs),
  | k).
  | 
  | In case there's less than k values in
  | a segment, the output value will be padded
  | by 0, and the corresponding output indices
  | will be padded by -1.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LengthsTopKOp<T,Context> {
    storage: OperatorStorage,
    context: Context,
    k:       i32,
    phantom: PhantomData<T>,
}

register_cpu_operator!{LengthsTopK, LengthsTopKOp<float, CPUContext>}

num_inputs!{LengthsTopK, 2}

num_outputs!{LengthsTopK, 2}

inputs!{LengthsTopK, 
    0 => ("DATA",        "Tensor of rank 1. First dimension must be equal to the sum of lengths"),
    1 => ("LENGTHS",     "Tensor of int32 lengths of rank 1")
}

outputs!{LengthsTopK, 
    0 => ("TopKValue",   "Output top k elements for each segment, with shape=(SIZE(lengths), k)"),
    1 => ("TopKIndices", "Output indices in DATA corresponding to value in TopKValue")
}

args!{LengthsTopK, 
    0 => ("k", "the number of top values to return for each segment, if the number of values is smaller than k, the values would be padded with 0 and indices would be padded with -1.")
}

input_tags!{
    LengthsTopKOp {
        XIn,
        YIn
    }
}

output_tags!{
    LengthsTopKOp {
        TopkValuesOut,
        TopkIndicesOut
    }
}
