crate::ix!();

/**
  | Replace the NaN (not a number) element
  | in the input tensor with argument `value`
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ReplaceNaNOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

register_cpu_operator!{ReplaceNaN, ReplaceNaNOp<CPUContext>}

num_inputs!{ReplaceNaN, 1}

num_outputs!{ReplaceNaN, 1}

inputs!{ReplaceNaN, 
    0 => ("input", "Input tensor"),
    1 => ("output", "Output tensor")
}

args!{ReplaceNaN, 
    0 => ("value (optional)", "the value to replace NaN, the default is 0")
}

identical_type_and_shape!{ReplaceNaN}

allow_inplace!{ReplaceNaN, vec![(0, 0)]}

should_not_do_gradient!{ReplaceNaN}
