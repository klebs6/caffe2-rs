crate::ix!();

/**
  | Relu takes one input data (Tensor) and
  | produces one output data (Tensor) where
  | the rectified linear function, y = xP(X
  | <= x) where X ~ N(0, 1), is applied to the
  | tensor elementwise.
  | 
  | Input: X, output: Y
  |
  */
pub type GeluOp<Context> = UnaryElementwiseWithArgsOp<
    TensorTypes<f32>,
    Context,
    GeluFunctor<Context>>;

num_inputs!{Gelu, 1}

num_outputs!{Gelu, 1}

args!{Gelu, 
    0 => ("fast_gelu", "If true, use y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3))).")
}

identical_type_and_shape!{Gelu}

inputs!{Gelu, 
    0 => ("X", "1D input tensor")
}

outputs!{Gelu, 
    0 => ("Y", "1D input tensor")
}

register_cpu_operator!{Gelu, GeluOp<CPUContext>}

pub const kFastCoeff: f32 = 0.044715;

pub struct GeluFunctor<Context> {
    fast_gelu: bool,

    phantom: PhantomData<Context>,
}

impl<Context> GeluFunctor<Context> {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
        todo!();
        /*
            : fast_gelu(op.GetSingleArgument<bool>("fast_gelu", false))
        */
    }
}
