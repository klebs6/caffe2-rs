crate::ix!();

pub struct CosineEmbeddingCriterionGradientOp<Context> {
    storage: OperatorStorage,
    context: Context,
    margin:  f32,
}

num_inputs!{CosineEmbeddingCriterionGradient, 3}

num_outputs!{CosineEmbeddingCriterionGradient, 1}

impl<Context> CosineEmbeddingCriterionGradientOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(float, "margin", margin_, 0.0)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& S = Input(0);
      auto& Y = Input(1);
      auto& dOutput = Input(2);

      auto* dS = Output(0, S.sizes(), at::dtype<float>());

      const float* Sdata = S.data<float>();
      const int* Ydata = Y.data<int>();
      const float* dOutput_data = dOutput.data<float>();
      float* dSdata = dS->template mutable_data<float>();
      for (int i = 0; i < S.numel(); ++i) {
        dSdata[i] = dOutput_data[i] *
            (Ydata[i] == 1 ? -1.f : static_cast<float>(Sdata[i] >= margin_));
      }
      return true;
        */
    }
}

