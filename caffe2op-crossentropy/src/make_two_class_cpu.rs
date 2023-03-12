crate::ix!();

impl MakeTwoClassOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);

      auto shape = X.sizes().vec();
      shape.push_back(2);
      int64_t N = X.numel();
      auto* Y = Output(0, shape, at::dtype<float>());
      const auto* Xdata = X.data<float>();
      auto* Ydata = Y->template mutable_data<float>();
      for (int64_t i = 0; i < N; ++i) {
        DCHECK_GE(Xdata[i], 0.0);
        DCHECK_LE(Xdata[i], 1.0);
        Ydata[i * 2] = 1.0 - Xdata[i];
        Ydata[i * 2 + 1] = Xdata[i];
      }
      return true;
        */
    }
}
