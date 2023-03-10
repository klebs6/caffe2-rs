crate::ix!();

impl AccuracyOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(PREDICTION);
      auto& label = Input(LABEL);

      CAFFE_ENFORCE_EQ(X.dim(), 2);
      int N = X.dim32(0);
      int D = X.dim32(1);
      CAFFE_ENFORCE_EQ(label.dim(), 1);
      CAFFE_ENFORCE_EQ(label.dim32(0), N);
      auto* Y = Output(0, vector<int64_t>(), at::dtype<float>());
      const auto* Xdata = X.data<float>();
      const auto* labelData = label.data<int>();
      const int top_k = top_k_;
      int correct = 0;

      // it's equivalent to using a stable sorting algorithm to sort the
      // classes (with their predictions as key) and then check whether
      // the label is within the first top_k slots.
      for (int i = 0; i < N; ++i) {
        auto label_i = labelData[i];
        auto label_pred = Xdata[i * D + label_i];
        int ngt = 1;
        for (int j = 0; j < D; ++j) {
          auto pred = Xdata[i * D + j];
          if ((pred > label_pred) || (pred == label_pred && j < label_i)) {
            if (++ngt > top_k) {
              break;
            }
          }
        }
        if (ngt <= top_k) {
          ++correct;
        }
      }
      CAFFE_ENFORCE_LE(correct, N);
      *(Y->template mutable_data<float>()) = static_cast<float>(correct) / N;

      return true;
        */
    }
}
