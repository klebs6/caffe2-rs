crate::ix!();

#[test] fn clip_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Clip",
        ["X"],
        ["Y"],
        min=20.0,
        max=60.0

    )

    workspace.FeedBlob("X", (np.random.randint(100, size=(5,5))).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X: [[45. 16. 59. 99. 48.]
     [12. 44. 46. 82. 28.]
     [ 1. 91. 18.  9. 71.]
     [24. 37. 61. 12. 81.]
     [36. 38. 30. 84. 40.]]

    Y: [[45. 20. 59. 60. 48.]
     [20. 44. 46. 60. 28.]
     [20. 60. 20. 20. 60.]
     [24. 37. 60. 20. 60.]
     [36. 38. 30. 60. 40.]]
    */
}

/**
  | This operator limits the given input
  | within an interval. The interval is
  | specified by the `min` and `max` arguments.
  | They default to numeric_limits::lowest()*
  | and numeric_limits::max()* respectively.
  | The clipping operation can be done in
  | an in-place fashion by using the same
  | output blob as the input blob.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/clip_op.cc
  |
  */
pub struct ClipOp<T, Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
    min:     T,
    max:     T,
}

num_inputs!{Clip, 1}

num_outputs!{Clip, 1}

inputs!{Clip, 
    0 => ("X", "*(Tensor`<float>`)* Input tensor within range [*numeric_limits::lowest()*, *numeric_limits::max()*].")
}

outputs!{Clip, 
    0 => ("Y", "*(Tensor`<float>`)* Output tensor clipped within range [`min`, `max`].")
}

args!{Clip, 
    0 => ("min", "*(type: float)* Minimum value, under which element is replaced by min (default=*numeric_limits::lowest()*)."),
    1 => ("max", "*(type: float)* Maximum value, under which element is replaced by max (default=*numeric_limits::max()*).")
}

identical_type_and_shape!{Clip}

inherit_onnx_schema!{Clip}

allow_inplace!{Clip, vec![(0, 0)]}

impl<T,Context> ClipOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            min_(std::numeric_limits<T>::lowest()),
            max_(T::max) 

        if (HasArgument("min")) {
          min_ = static_cast<T>(this->template GetSingleArgument<float>("min", 0));
        }
        if (HasArgument("max")) {
          max_ = static_cast<T>(this->template GetSingleArgument<float>("max", 0));
        }
        */
    }
}

pub struct ClipGradientOp<T, Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    min: T,
    max: T,

    // Input: Y, dY; Output: dX
}

num_inputs!{ClipGradient, 2}

num_outputs!{ClipGradient, 1}

allow_inplace!{ClipGradient, vec![(1, 0)]}

impl<T, Context> ClipGradientOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            min_(std::numeric_limits<T>::lowest()),
            max_(T::max) 

        if (HasArgument("min")) {
          min_ = static_cast<T>(this->template GetSingleArgument<float>("min", 0));
        }
        if (HasArgument("max")) {
          max_ = static_cast<T>(this->template GetSingleArgument<float>("max", 0));
        }
        */
    }
}

impl ClipGradientOp<f32, CPUContext> {

    #[inline] pub fn run_on_deviceA(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);

      auto* Y = Output(0, X.sizes(), at::dtype<float>());
      EigenVectorMap<float>(Y->template mutable_data<float>(), Y->numel()) =
          ConstEigenVectorMap<float>(X.data<float>(), X.numel())
              .cwiseMax(min_)
              .cwiseMin(max_);
      return true;
        */
    }
    
    #[inline] pub fn run_on_deviceB(&mut self) -> bool {
        
        todo!();
        /*
            auto& Y = Input(0);
      auto& dY = Input(1);

      CAFFE_ENFORCE_GE(Y.numel(), 0);
      CAFFE_ENFORCE_EQ(dY.numel(), Y.numel());
      auto* dX = Output(0, Y.sizes(), at::dtype<float>());
      const float* Ydata = Y.data<float>();
      const float* dYdata = dY.data<float>();
      float* dXdata = dX->template mutable_data<float>();
      for (int i = 0; i < Y.numel(); ++i) {
        dXdata[i] = dYdata[i] * (Ydata[i] > min_ && Ydata[i] < max_);
      }
      return true;
        */
    }
}

register_cpu_operator!{
    Clip, 
    ClipOp<f32, CPUContext>
}

register_cpu_gradient_operator!{
    ClipGradient, 
    ClipGradientOp<f32, CPUContext>
}

pub struct GetClipGradient;

impl GetGradientDefs for GetClipGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "ClipGradient", "",
            vector<string>{O(0), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{Clip, GetClipGradient}
