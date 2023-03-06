crate::ix!();

impl MarginRankingCriterionOp<CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X1 = Input(0);
      auto& X2 = Input(1);
      auto& Y = Input(2);

      CAFFE_ENFORCE_EQ(
          X1.numel(),
          X2.numel(),
          "The two inputs for computing ranking loss should have the same size.");
      CAFFE_ENFORCE_EQ(
          X1.numel(), Y.numel(), "The input and label should have the same size.");
      auto* loss = Output(0, X1.sizes(), at::dtype<float>());

      const float* X1data = X1.data<float>();
      const float* X2data = X2.data<float>();
      const int* Ydata = Y.data<int>();
      float* output = loss->template mutable_data<float>();
      for (int i = 0; i < X1.numel(); ++i) {
        output[i] = std::max(-Ydata[i] * (X1data[i] - X2data[i]) + margin_, 0.f);
      }
      return true;
        */
    }
}
