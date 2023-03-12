crate::ix!();

impl LabelCrossEntropyOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
      auto& label = Input(1);

      int N, D;
      if (X.dim() > 1) {
        N = X.dim32(0);
        D = X.size_from_dim(1);
      } else {
        N = 1;
        D = X.dim32(0);
      }
      CAFFE_ENFORCE(
          (label.dim() == 1) || (label.dim() == 2 && label.dim32(1) == 1));
      CAFFE_ENFORCE_EQ(label.dim32(0), N);
      auto* Y = Output(0, {N}, at::dtype<float>());
      const auto* Xdata = X.data<float>();
      const auto* labelData = label.data<int>();
      auto* Ydata = Y->template mutable_data<float>();
      CAFFE_ENFORCE(
          (ConstEigenVectorArrayMap<int>(labelData, N) < D).all() &&
              (ConstEigenVectorArrayMap<int>(labelData, N) >= 0).all(),
          "Label seems to be outside of supported range. Supported labels are in "
          "range [0,",
          D,
          ")");
      for (int i = 0; i < N; ++i) {
        Ydata[i] = -log(std::max(Xdata[i * D + labelData[i]], kLOG_THRESHOLD()));
      }
      return true;
        */
    }
}
