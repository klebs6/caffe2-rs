crate::ix!();

/**
  | Applies gated linear unit to the input
  | Tensor X.
  | 
  | The output Y is half the size of the input
  | X, so if the shape of X is [d1, d2, ...,
  | N] shape of
  | 
  | Y will be [d1, d2, ..., dn/2] and Y(:dn-1,
  | i) = GLU(X(:dn-1, i), X(:dn-1, i+N/2))
  | = X(dn-1, i) sigmoid(X(dn-1, i+N/2))
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct GluOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    dim:     i32,
    phantom: PhantomData<T>,
}

num_inputs!{Glu, 1}

num_outputs!{Glu, 1}

inputs!{Glu, 
    0 => ("X", "1D input tensor")
}

outputs!{Glu, 
    0 => ("Y", "1D output tensor")
}

register_cpu_operator!{Glu, GluOp<f32, CPUContext>}
