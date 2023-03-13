crate::ix!();

/**
  | The *LeakyRelu* op takes one input tensor
  | $X$ and an argument $alpha$, and produces
  | one output tensor $Y$ of the same shape
  | as $X.$ The op performs the element wise
  | leaky relu operation, defined as
  | 
  | $$y=LeakyRelu(x) =\begin{cases}\alpha
  | x & x < 0\\x & otherwise\end{cases}$$
  | 
  | The default value of *alpha* is 0.01.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/leaky_relu_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/leaky_relu_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LeakyReluOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    alpha:   T,
}

register_cpu_operator!{
    LeakyRelu, 
    LeakyReluOp<f32, CPUContext>
}

num_inputs!{LeakyRelu, 1}

num_outputs!{LeakyRelu, 1}

inputs!{LeakyRelu, 
    0 => ("X",      "Input tensor of data to be operated on.")
}

outputs!{LeakyRelu, 
    0 => ("Y",      "Output tensor, calculated as described above.")
}

args!{LeakyRelu, 
    0 => ("alpha",  "*(type: float; default: 0.01)* Coefficient of leakage.")
}

identical_type_and_shape!{LeakyRelu}

cost_inference_function!{LeakyRelu, PointwiseCostInference::<2>}

allow_inplace!{LeakyRelu, vec![(0, 0)]}
