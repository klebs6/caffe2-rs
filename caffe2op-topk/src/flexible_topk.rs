crate::ix!();

/**
  | Given two tensors: X and K, retrieve
  | the top
  | 
  | K[..., 1] elements from X on the last
  | dimension.
  | 
  | X is an input tensor of shape [a_1, a_2,
  | ..., a_n, r].
  | 
  | K is an input tensor of shape [a_1, a_2,
  | ..., a_n, 1], where for each element,
  | r >= K[..., 1] > 0
  | 
  | Output two outputs:
  | 
  | -Flatten values tensor of shape [ \sum_i
  | K[i, 1] ] which contains the values of
  | the top K[..., 1] elements along the
  | last dimension
  | 
  | -Flatten indices tensor of shape [ \sum_i
  | K[i, 1] ] which contains the indices
  | of the top K[..., 1] elements, flatten
  | indices from the input tensor).
  | 
  | These two outputs should be used with
  | the input K, so that we know which indices
  | in X are picked.
  | 
  | Given two equivalent values, this operator
  | uses the indices along the last dim-
  | ension as a tiebreaker. That is, the
  | element with the lower index will appear
  | first.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct FlexibleTopKOp<T, Context> {
    context: Context,
    phantom: PhantomData<T>,
}

num_inputs!{FlexibleTopK, 2}

num_outputs!{FlexibleTopK, 2}

inputs!{FlexibleTopK, 
    0 => ("X", "Tensor of shape [a_1, a_2, ..., a_n, r]"),
    1 => ("K", "Tensor of shape [a_1, a_2, ..., a_n, 1]")
}

outputs!{FlexibleTopK, 
    0 => ("Flatten values", "Tensor of shape [ \\sum_i K[i, 1] ] containing top K[..., 1] values from the input tensor"),
    1 => ("Flatten indices", "Tensor of shape [ \\sum_i K[i, 1] ] containing the indices into the flatten input")
}

register_cpu_operator!{
    FlexibleTopK, 
    FlexibleTopKOp<float, CPUContext>
}
