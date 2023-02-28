crate::ix!();

///----------------
#[test] fn relu_functor_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
      "Relu",
      ["X"],
      ["Y"]
      )

    workspace.FeedBlob("X", np.random.randn(4, 4).astype(np.float32)) // NCHW
    print("X:\n", workspace.FetchBlob("X"), "\n")

    workspace.RunOperatorOnce(op)
    print("Y:\n", workspace.FetchBlob("Y"))

    X:
     [[-1.4655551   0.64575136  0.7921748   0.4150579 ]
     [ 0.41085166 -0.2837964   0.9881425  -1.9300346 ]
     [ 0.39705405  0.44639114  0.9940703   0.2926532 ]
     [-0.6726489   0.01330667  1.101319    0.33858967]]

    Y:
     [[0.         0.64575136 0.7921748  0.4150579 ]
     [0.41085166 0.         0.9881425  0.        ]
     [0.39705405 0.44639114 0.9940703  0.2926532 ]
     [0.         0.01330667 1.101319   0.33858967]]
    */
}

/**
 | Applies rectified linear unit operation to the
 | input data element-wise. The Relu operation takes
 | one input $X$, produces one output $Y$, and is
 | defined as:
 |
 | $$Y = max(0,X)$$
 |
 | Github Links:
 | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/relu_op.h
 | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/relu_op.cc
 |
 */
pub struct ReluFunctor<Context> {
    
    /**
      | Input: X
      | 
      | output: Y
      |
      */
    phantom: PhantomData<Context>,
}

num_inputs!{Relu, 1}

num_outputs!{Relu, 1}

inputs!{Relu, 
    0 => ("X", "1D input tensor")
}

outputs!{Relu, 
    0 => ("Y", "1D output tensor with same shape as input")
}

identical_type_and_shape!{Relu}

cost_inference_function!{Relu, CostInferenceForRelu }

allow_inplace!{Relu, vec![(0, 0)]}

inherit_onnx_schema!{Relu}

impl ReluFunctor<CPUContext> {

    #[inline] pub fn invoke<T>(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut CPUContext) -> bool {
    
        todo!();
        /*
            EigenVectorMap<T>(Y, N) = ConstEigenVectorMap<float>(X, N).cwiseMax(T(0));
      return true;
        */
    }
    
    #[cfg(caffe2_use_accelerate)]
    #[inline] pub fn invoke_f32(&self, 
        n:       i32,
        x:       *const f32,
        y:       *mut f32,
        context: *mut CPUContext) -> bool {
        
        todo!();
        /*
          const float zero = 0.0f;
          vDSP_vthres(X, 1, &zero, Y, 1, N);
          return true;
        */
    }
}

/**
  | ReluGradient takes both Y and dY and
  | uses this to update dX according to the
  | chain rule and derivatives of the rectified
  | linear function.
  |
  */
pub struct ReluGradientFunctor<Context> {

    /**
      | Input: Y, dY
      | 
      | output: dX
      |
      */
    phantom: PhantomData<Context>,
}

num_inputs!{ReluGradient, 2}

num_outputs!{ReluGradient, 1}

identical_type_and_shape_of_input!{ReluGradient, 1}

allow_inplace!{ReluGradient, vec![(1, 0)]}

impl ReluGradientFunctor<CPUContext> {

    #[inline] pub fn forward<T>(&self, 
        y_dims:  &Vec<i32>,
        dy_dims: &Vec<i32>,
        y:       *const T,
        dy:      *const T,
        dx:      *mut T,
        context: *mut CPUContext) -> bool {
    
        todo!();
        /*
            const int size = std::accumulate(
          Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
      EigenVectorArrayMap<T>(dX, size) =
          (ConstEigenVectorArrayMap<T>(Y, size) > T(0))
              .select(ConstEigenVectorArrayMap<T>(dY, size), T(0));
      return true;
        */
    }
}

#[inline] pub fn cost_inference_for_relu(
    def:   &OperatorDef, 
    input: &Vec<TensorShape>) -> OpSchemaCost {
    
    todo!();
    /*
        struct OpSchema::Cost cost = PointwiseCostInference<0>(def, in);
      cost.params_bytes = 0;
      return cost;
    */
}

register_cpu_operator!{Relu,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        ReluFunctor<CPUContext>>}

register_cpu_gradient_operator!{ReluGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        ReluGradientFunctor<CPUContext>>}

pub struct GetReluGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetReluGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            def_.type() + "Gradient",
            "",
            std::vector<std::string>{O(0), GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Relu, GetReluGradient}
