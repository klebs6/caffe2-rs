crate::ix!();

/**
  | ThresholdedRelu takes one input data
  | (Tensor) and produces one output data
  | (Tensor) where the rectified linear
  | function, y = x for x > alpha, y = 0 otherwise,
  | is applied to the tensor elementwise.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ThresholdedReluOp<T,Context> {

    storage: OperatorStorage,
    context: Context,

    alpha:   T,

    /*
      | Input: X,
      | 
      | output: Y
      |
      */
}

register_cpu_operator!{ThresholdedRelu, ThresholdedReluOp<float, CPUContext>}

num_inputs!{ThresholdedRelu, 1}

num_outputs!{ThresholdedRelu, 1}

inputs!{ThresholdedRelu, 
    0 => ("X", "1D input tensor")
}

outputs!{ThresholdedRelu, 
    0 => ("Y", "1D input tensor")
}

args!{ThresholdedRelu, 
    0 => ("alpha", "(float) defaults to 1.0.")
}

identical_type_and_shape!{ThresholdedRelu}

cost_inference_function!{ThresholdedRelu, 
    PointwiseCostInference::<2>
}

allow_inplace!{ThresholdedRelu, vec![(0, 0)]}

impl<T,Context> ThresholdedReluOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        alpha_ = this->template GetSingleArgument<T>("alpha", 1.0);
        */
    }
}
