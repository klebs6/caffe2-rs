crate::ix!();

impl MultiClassAccuracyOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(PREDICTION);
      auto& label = Input(LABEL);

      DCHECK_EQ(X.dim(), 2);
      // amount, number of instances
      int N = X.dim32(0);
      // dimension, number of classes
      int D = X.dim32(1);
      DCHECK_EQ(label.dim(), 1);
      DCHECK_EQ(label.dim32(0), N);
      auto* Y0 = Output(0, {D}, at::dtype<float>());
      auto* Y1 = Output(1, {D}, at::dtype<int>());

      const auto* Xdata = X.data<float>();
      const auto* labeldata = label.data<int>();
      auto* accuracies = Y0->template mutable_data<float>();
      auto* amounts = Y1->template mutable_data<int>();
      std::fill(accuracies, accuracies + D, 0);
      std::fill(amounts, amounts + D, 0);

      for (int i = 0; i < N; ++i) {
        float maxval = std::numeric_limits<float>::lowest();
        int maxid = 0;
        for (int j = 0; j < D; ++j) {
          if (Xdata[i * D + j] > maxval) {
            maxval = Xdata[i * D + j];
            maxid = j;
          }
        }
        int labelid = labeldata[i];
        DCHECK_LT(labelid, D);
        if (maxid == labelid) {
          accuracies[labelid]++;
        }
        amounts[labelid]++;
      }

      for (int i = 0; i < D; ++i) {
        int amount = amounts[i];
        if (amount) {
          accuracies[i] /= amount;
        }
      }

      return true;
        */
    }
}
