crate::ix!();

/**
  | Generates an output tensor within the
  | half-open interval $[start, stop)$
  | (the interval including start but excluding
  | stop).
  | 
  | - The `start` input is optional, and
  | defaults to 0 when not set.
  | 
  | - The `step` input is optional, and defaults
  | to 1 when not set.
  | 
  | - The type of the `output` tensor is determined
  | by the types of inputs used.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct RangeOp<Context> {
    storage: OperatorStorage,
    context: Context,

    /// local CPU tensor for copying constants.
    local:   Tensor, // default = CPU
}

num_inputs!{Range, (1,3)}

num_outputs!{Range, 1}

inputs!{Range, 
    0 => ("start",  "(*Tensor*): [OPTIONAL] scalar or 1-element tensor containing the start of the interval (inclusive) (default=0)"),
    1 => ("stop",   "(*Tensor*): scalar or 1-element tensor containing the end of the interval (exclusive)"),
    2 => ("step",   "(*Tensor*): [OPTIONAL] scalar or 1-element tensor specifying the spacing between values (default=1)")
}

outputs!{Range, 
    0 => ("output", "(*Tensor*): 1D tensor of same type as inputs that contains the sequence")
}

register_cpu_operator!{Range, RangeOp<CPUContext>}

no_gradient!{Range}
