crate::ix!();

/**
  | Softplus takes one input data tensor
  | $X$ and produces one output data tensor
  | $Y,$ where the softplus function, $y
  | = ln(e^x + 1)$, is applied to $X$ elementwise.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softplus_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softplus_op.cc
  |
  */
pub struct SoftplusOp<DataType> {

    storage:         OperatorStorage,
    context:         CPUContext,

    /**
      | Input: X
      | 
      | output: Y
      |
      */
    phantomDataType: PhantomData<DataType>,
}

num_inputs!{Softplus, 1}

num_outputs!{Softplus, 1}

inputs!{Softplus, 
    0 => ("X", "Input data blob to be operated on.")
}

outputs!{Softplus, 
    0 => ("Y", "Output data blob with same shape as input.")
}

identical_type_and_shape!{Softplus}

allow_inplace!{Softplus, vec![(0, 0)]}

inherit_onnx_schema!{Softplus}

impl SoftplusOp<DataType> {

    fn run_on_device() -> bool {

        todo!();
        /*
        auto& X = Input(0);

        auto* Y = Output(0, X.sizes(), at::dtype<float>());

        EigenVectorMap<float>(Y->template mutable_data<float>(), X.numel()) =
            (ConstEigenVectorMap<float>(X.data<float>(), X.numel()).array().exp() +
             1.0f)
            .log();
        return true;
        */
    }
}
