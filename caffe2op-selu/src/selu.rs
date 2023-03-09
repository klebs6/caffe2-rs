crate::ix!();

/**
  | The *Selu* op takes one input tensor
  | $X$, an argument $alpha$, an argument
  | $scale$, and produces one output tensor
  | $Y$ of the same shape as $X.$ The op performs
  | the element wise *Selu* operation,
  | defined as
  | 
  | $$y=selu(x) =\begin{cases}scale
  | (\alpha e^{x} - \alpha) & x < 0\\scale
  | * x & otherwise\end{cases}$$
  | 
  | The default value of *alpha* is 1.6732632423543772848170429916717
  | and the default value of *scale* is 1.0507009873554804934193349852946.
  | See [Self-Normalizing Neural
  | 
  | Networks](https://arxiv.org/abs/1706.02515)
  | for more information.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/selu_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/selu_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SeluOp<T,Context> {

    storage: OperatorStorage,
    context: Context,

    alpha:   T,
    lambda:  T,

    /*
      | Input: X;
      | 
      | output: Y
      |
      */
}

num_inputs!{Selu, 1}

num_outputs!{Selu, 1}

inputs!{Selu, 
    0 => ("X", "Input tensor of data to be operated on.")
}

outputs!{Selu, 
    0 => ("Y", "Output tensor with same shape as input.")
}

args!{Selu, 
    0 => ("alpha", "*(type: float; default: 1.673263~)* Alpha constant in equation."),
    1 => ("scale", "*(type: float; default: 1.050700~; must be > 1.0)* Scale constant in equation.")
}

identical_type_and_shape!{Selu}

allow_inplace!{Selu, vec![(0, 0)]}

inherit_onnx_schema!{Selu}

impl<T,Context> SeluOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        alpha_ = this->template GetSingleArgument<T>(
            "alpha", 1.6732632423543772848170429916717f);
        lambda_ = this->template GetSingleArgument<T>(
            "scale", 1.0507009873554804934193349852946f);
        // In the paper "scale" is named "lambda", but "lambda" is a reserved
        // keyword in python
        CAFFE_ENFORCE_GT(lambda_, 1.0);
        */
    }
}
