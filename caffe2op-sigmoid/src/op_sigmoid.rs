crate::ix!();

///--------------------

#[test] fn sigmoid_functor_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Sigmoid",
        ["X"],
        ["Y"]
    )

    workspace.FeedBlob("X", np.random.randn(5).astype(np.float32))
    print("input:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("sigmoid:", workspace.FetchBlob("Y"))

    input: [ 1.5744036   0.31632107  1.7842269   1.4450722  -2.1726978 ]
    sigmoid: [0.8284105  0.57842743 0.85621804 0.80923885 0.10222916]
    */
}

/**
  | Apply the Sigmoid function element-wise
  | to the input tensor. This is often used
  | as a non-linear activation function
  | in a neural network. The sigmoid function
  | is defined as:
  | 
  | $$Sigmoid(x) = \frac{1}{1+\exp(-x)}$$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sigmoid_op.cc
  |
  */
pub struct SigmoidFunctor<Context> {
    
    // Input: X, output: Y
    phantom: PhantomData<Context>,
}

num_inputs!{Sigmoid, 1}

num_outputs!{Sigmoid, 1}

inputs!{Sigmoid, 
    0 => ("X", "*(type: Tensor`<float>`)* Input tensor.")
}

outputs!{Sigmoid, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor.")
}

identical_type_and_shape!{Sigmoid}

allow_inplace!{Sigmoid, vec![(0, 0)]}

inherit_onnx_schema!{Sigmoid}

impl SigmoidFunctor<CPUContext> {
    
    #[inline] pub fn invoke<T>(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut CPUContext) -> bool {
    
        todo!();
        /*
          EigenVectorArrayMap<T>(Y, N) =
          T(1) / (T(1) + (-ConstEigenVectorArrayMap<T>(X, N)).exp());
          return true;
        */
    }
}

/**
  | SigmoidGradient takes both Y and dY
  | and uses this to update dX according
  | to the chain rule and derivatives of
  | the sigmoid function.
  |
  */
pub struct SigmoidGradientFunctor<Context> {

    /**
      | Input: Y, dY
      | 
      | output: dX
      |
      */
    phantom: PhantomData<Context>,
}

register_cpu_operator!{Sigmoid,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        SigmoidFunctor<CPUContext>>}

num_inputs!{SigmoidGradient, 2}

num_outputs!{SigmoidGradient, 1}

identical_type_and_shape_of_input!{SigmoidGradient, 1}

allow_inplace!{SigmoidGradient, vec![(1, 0)]}
