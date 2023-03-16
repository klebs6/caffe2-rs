crate::ix!();

/**
  | Scale takes one input data (Tensor)
  | and produces one output data (Tensor)
  | whose value is the input data tensor
  | scaled element-wise.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ScaleOp<Context> {
    storage: OperatorStorage,
    context: Context,
    scale:   f32,
}

register_cpu_operator!{Scale, ScaleOp<CPUContext>}

num_inputs!{Scale, 1}

num_outputs!{Scale, 1}

args!{Scale, 
    0 => ("scale", "(float, default 1.0) the scale to apply.")
}

identical_type_and_shape!{Scale}

allow_inplace!{Scale, vec![(0, 0)]}
