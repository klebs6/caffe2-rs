crate::ix!();

impl MarginRankingCriterionGradientOp<CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X1 = Input(0);
      auto& X2 = Input(1);
      auto& Y = Input(2);
      auto& dLoss = Input(3);

      auto* dX1 = Output(0, X1.sizes(), at::dtype<float>());
      auto* dX2 = Output(1, X2.sizes(), at::dtype<float>());

      const float* X1data = X1.data<float>();
      const float* X2data = X2.data<float>();
      const int* Ydata = Y.data<int>();
      const float* dLoss_data = dLoss.data<float>();

      float* dX1_data = dX1->template mutable_data<float>();
      float* dX2_data = dX2->template mutable_data<float>();
      for (int i = 0; i < X1.numel(); ++i) {
        auto dist = -Ydata[i] * (X1data[i] - X2data[i]) + margin_;
        if (dist < 0.f) {
          dX1_data[i] = dX2_data[i] = 0.f;
        } else {
          dX1_data[i] = -Ydata[i] * dLoss_data[i];
          dX2_data[i] = Ydata[i] * dLoss_data[i];
        }
      }
      return true;
        */
    }
}

register_cpu_operator!{
    MarginRankingCriterion,
    MarginRankingCriterionOp<CPUContext>
}

register_cpu_operator!{
    MarginRankingCriterionGradient,
    MarginRankingCriterionGradientOp<CPUContext>
}
