crate::ix!();

/**
  | Applies hard sigmoid operation to the
  | input data element-wise.
  | 
  | The HardSigmoid operation takes one
  | input $X$, produces one output $Y$,
  | and is defined as:
  | 
  | $$Y = max(0,min(1,x * alpha + beta))$$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/hard_sigmoid_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/hard_sigmoid_op.cc
  |
  */
pub struct HardSigmoidFunctor<Context> {
    alpha:   f32,
    beta:    f32,

    /**
      | Input: X
      | 
      | Output: Y
      |
      */
    phantom: PhantomData<Context>,
}

num_inputs!{HardSigmoid, 1}

num_outputs!{HardSigmoid, 1}

inputs!{HardSigmoid, 
    0 => ("X", "1D input tensor")
}

outputs!{HardSigmoid, 
    0 => ("Y", "1D output tensor with same shape as input")
}

args!{HardSigmoid, 
    0 => ("alpha", "float: the slope of the function. Defaults to 0.2"),
    1 => ("beta", "float: the bias value of the function. Defaults to 0.5")
}

identical_type_and_shape!{HardSigmoid}

allow_inplace!{HardSigmoid, vec![(0, 0)]}

cost_inference_function!{HardSigmoid, CostInferenceForHardSigmoid }

inherit_onnx_schema!{HardSigmoid}

impl<Context> HardSigmoidFunctor<Context> {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
        todo!();
        /*
            : alpha(op.GetSingleArgument<float>("alpha", 0.2f)),
            beta(op.GetSingleArgument<float>("beta", 0.5f))
        */
    }
}
