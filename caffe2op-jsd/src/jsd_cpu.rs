crate::ix!();

impl BernoulliJSDOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0); // predicted probabilities
      auto& T = Input(1); // target probabilities
      int N = X.numel();
      CAFFE_ENFORCE_EQ(T.numel(), N);
      auto* L = Output(0, X.sizes(), at::dtype<float>()); // JSD loss output
      auto* x_data = X.data<float>();
      auto* t_data = T.data<float>();
      auto* l_data = L->template mutable_data<float>();
      for (int i = 0; i < N; i++) {
        auto p_mdl = x_data[i];
        auto p_emp = t_data[i];
        auto p_avg = (p_mdl + p_emp) / 2.;
        auto jsd = entropy(p_avg) - (entropy(p_mdl) + entropy(p_emp)) / 2.;
        l_data[i] = jsd;
      }
      return true;
        */
    }
}
