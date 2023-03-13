crate::ix!();

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
