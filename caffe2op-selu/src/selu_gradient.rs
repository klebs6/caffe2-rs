crate::ix!();

/**
  | SeluGradient takes both Y and dY and
  | uses this to update dX according to the
  | chain rule and derivatives of the selu
  | function.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SeluGradientOp<T,Context> {

    storage: OperatorStorage,
    context: Context,

    alpha:   T,
    lambda:  T,

    /*
      | Input: Y, dY;
      | 
      | output: dX
      |
      */
}

num_inputs!{SeluGradient, 2}

num_outputs!{SeluGradient, 1}

inputs!{SeluGradient, 
    0 => ("Y", "input tensor"),
    1 => ("dY", "input tensor")
}

args!{SeluGradient, 
    0 => ("alpha", "(float) default to 1.6732~; affects the activation function itself. This should go with the weight initialization in the paper.  See https://arxiv.org/abs/1706.02515 "),
    1 => ("scale", "(float) default to 1.0507~; affects the activation function itself.")
}

allow_inplace!{SeluGradient, vec![(1, 0)]}

impl<T,Context> SeluGradientOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        alpha_ = this->template GetSingleArgument<T>(
            "alpha", 1.6732632423543772848170429916717f);
        lambda_ = this->template GetSingleArgument<T>(
            "scale", 1.0507009873554804934193349852946f);
        CAFFE_ENFORCE_GT(lambda_, 1.0);
        */
    }
}
