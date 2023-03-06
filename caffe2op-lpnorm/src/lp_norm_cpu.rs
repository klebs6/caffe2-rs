crate::ix!();

impl LpNormOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);

      auto* norm = Output(0, {1}, at::dtype<float>());
      const float* X_data = X.data<float>();
      const float size = average_ ? (float)X.numel() : 1.0f;
      CAFFE_ENFORCE_GT(size, 0);
      if (p_ == 1) {
        *(norm->template mutable_data<float>()) =
            (ConstEigenVectorMap<float>(X_data, X.numel()).array()).abs().sum() /
            size;
        // L1(x) = sum(|x|), L1_average(x) = sum(\x\) / x.size()
      } else if (p_ == 2) {
        *(norm->template mutable_data<float>()) =
            (ConstEigenVectorMap<float>(X_data, X.numel()).array()).square().sum() /
            size;
        // L2(x) = (sum(|x|^2)), L2_average(x) = sum(|x|^2) / x.size()
      }
      return true;
        */
    }
}
