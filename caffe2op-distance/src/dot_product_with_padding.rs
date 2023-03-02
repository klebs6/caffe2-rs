crate::ix!();

/**
  | Given two input float tensors X, Y with
  | different shapes and produces one output
  | float tensor of the dot product between
  | X and Y.
  | 
  | We currently support two kinds of strategies
  | to achieve this.
  | 
  | Before doing normal dot_product
  | 
  | 1) pad the smaller tensor (using pad_value)
  | to the same shape as the other one.
  | 
  | 2) replicate the smaller tensor to the
  | same shape as the other one.
  | 
  | Note the first dimension of X, Y must
  | be equal. Only the second dimension
  | of X or Y can be padded.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct DotProductWithPaddingOp<T, Context> {

    storage: OperatorStorage,
    context: Context,

    pad_value: f32,
    replicate: bool,

    phantom: PhantomData<T>,
}

num_inputs!{DotProductWithPadding, 2}

num_outputs!{DotProductWithPadding, 1}

inputs!{DotProductWithPadding, 
    0 => ("X", "1D or 2D input tensor"),
    1 => ("Y", "1D or 2D input tensor")
}

outputs!{DotProductWithPadding, 
    0 => ("Z", "1D output tensor")
}

args!{DotProductWithPadding, 
    0 => ("pad_value", "the padding value for tensors with smaller dimension"),
    1 => ("replicate", "whether to replicate the smaller tensor or not")
}

identical_type_and_shape_of_input_dim!{DotProductWithPadding, (0, 0)}

register_cpu_operator!{
    DotProductWithPadding,
    DotProductWithPaddingOp<float, CPUContext>
}

impl<T,Context> DotProductWithPaddingOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            pad_value_(this->template GetSingleArgument<float>("pad_value", 0.0)),
            replicate_(this->template GetSingleArgument<bool>("replicate", false))
        */
    }
}

input_tags!{
    DotProductWithPaddingOp {
        XIn,
        YIn
    }
}

output_tags!{
    DotProductWithPaddingOp {
        DotOut
    }
}
