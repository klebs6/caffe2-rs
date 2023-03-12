crate::ix!();

#[test] fn leaky_relu_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "LeakyRelu",
        ["X"],
        ["Y"],
        alpha=0.01
    )

    workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32))
    print("X:\n", workspace.FetchBlob("X"), "\n")

    workspace.RunOperatorOnce(op)
    print("Y:\n", workspace.FetchBlob("Y"))

    X:
     [[-0.91060215  0.09374836  2.1429708 ]
     [-0.748983    0.19164062 -1.5130422 ]
     [-0.29539835 -0.8530696   0.7673204 ]]

    Y:
     [[-0.00910602  0.09374836  2.1429708 ]
     [-0.00748983  0.19164062 -0.01513042]
     [-0.00295398 -0.0085307   0.7673204 ]]
    */
}

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

    alpha: T,
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

impl<T, Context> LeakyReluOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...), alpha_(0.01) 

        if (HasArgument("alpha")) {
          alpha_ = static_cast<T>(
              this->template GetSingleArgument<float>("alpha", 0.01));
        }
        */
    }
}

impl LeakyReluOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);

      auto* Y = Output(0, X.sizes(), at::dtype<float>());
      ConstEigenVectorMap<float> Xvec(X.template data<float>(), X.numel());
      EigenVectorMap<float> Yvec(Y->template mutable_data<float>(), Y->numel());
      Yvec = Xvec.cwiseMax(0.f) + Xvec.cwiseMin(0.f) * alpha_;
      return true;
        */
    }
}

///--------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LeakyReluGradientOp<T, Context> {
    storage: OperatorStorage,
    context: Context,

    alpha: T,
}

num_inputs!{LeakyReluGradient, 2}

num_outputs!{LeakyReluGradient, 1}

args!{LeakyReluGradient, 
    0 => ("alpha", "Coefficient of leakage")
}

identical_type_and_shape_of_input!{LeakyReluGradient, 1}

allow_inplace!{LeakyReluGradient, vec![(1, 0)]}

inherit_onnx_schema!{LeakyReluGradient}

impl<T, Context> LeakyReluGradientOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...), alpha_(0.01) 

        if (HasArgument("alpha")) {
          alpha_ = static_cast<T>(
              this->template GetSingleArgument<float>("alpha", 0.01));
        }
        */
    }
}

impl LeakyReluGradientOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& Y = Input(0);
      const auto& dY = Input(1);

      auto* dX = Output(0, Y.sizes(), at::dtype<float>());
      CAFFE_ENFORCE_EQ(Y.numel(), dY.numel());
      ConstEigenVectorMap<float> Yvec(Y.template data<float>(), Y.numel());
      ConstEigenVectorMap<float> dYvec(dY.template data<float>(), dY.numel());
      EigenVectorMap<float> dXvec(dX->template mutable_data<float>(), dX->numel());
      Eigen::VectorXf gtZero = (Yvec.array() >= 0.0f).cast<float>();
      dXvec = dYvec.array() * gtZero.array() -
          dYvec.array() * (gtZero.array() - 1.0f) * alpha_;
      return true;
        */
    }
}

register_cpu_operator!{
    LeakyRelu, 
    LeakyReluOp<f32, CPUContext>
}

register_cpu_operator!{
    LeakyReluGradient,
    LeakyReluGradientOp<f32, CPUContext>
}

pub struct GetLeakyReluGradient;

impl GetGradientDefs for GetLeakyReluGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "LeakyReluGradient",
            "",
            vector<string>{O(0), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{LeakyRelu, GetLeakyReluGradient}
