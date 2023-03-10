crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ClipGradientOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    min:     T,
    max:     T,

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
