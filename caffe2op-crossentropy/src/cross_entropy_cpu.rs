crate::ix!();

impl CrossEntropyOp<f32, CPUContext> {

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
          (label.dim() == 1) || (label.dim() == 2 && label.dim32(1) == D));
      CAFFE_ENFORCE_EQ(label.dim32(0), N);
      auto* Y = Output(0, vector<int64_t>{N}, at::dtype<float>());
      const float* Xdata = X.data<float>();
      const float* labelData = label.data<float>();
      auto* Ydata = Y->template mutable_data<float>();
      CAFFE_ENFORCE(
          (ConstEigenArrayMap<float>(labelData, D, N) <= 1.0f).all() &&
              (ConstEigenArrayMap<float>(labelData, D, N) >= 0.0f).all(),
          "Soft label seems incorrect: label value should be a probability ",
          "between 0 and 1.0. You may be using the wrong cross entropy operator; ",
          "use LabelCrossEntropy if the labels are integers whose values are at ",
          "most the number of classes, ",
          D,
          ".");
      EigenArrayMap<float>(Ydata, 1, N) =
          -(ConstEigenArrayMap<float>(labelData, D, N) *
            ConstEigenArrayMap<float>(Xdata, D, N).cwiseMax(kLOG_THRESHOLD()).log())
               .colwise()
               .sum();
      return true;
        */
    }
}
