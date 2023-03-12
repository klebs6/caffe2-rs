crate::ix!();

impl CosineEmbeddingCriterionOp<CPUContext> {

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

      CAFFE_ENFORCE(
          S.numel() == Y.numel(),
          "The embedding and label should have the same size.");
      auto* output = Output(0, S.sizes(), at::dtype<float>());

      const float* Sdata = S.data<float>();
      const int* Ydata = Y.data<int>();
      float* output_data = output->template mutable_data<float>();
      for (int i = 0; i < S.numel(); ++i) {
        output_data[i] =
            Ydata[i] == 1 ? (1.f - Sdata[i]) : std::max(0.f, Sdata[i] - margin_);
      }
      return true;
        */
    }
}
