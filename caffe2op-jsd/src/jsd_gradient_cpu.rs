crate::ix!();

impl BernoulliJSDGradientOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
          auto& go = Input(0);
          auto& X = Input(1);
          auto& T = Input(2);

          int N = X.numel();
          auto* gi = Output(0, X.sizes(), at::dtype<float>());
          auto* go_data = go.data<float>();
          auto* x_data = X.data<float>();
          auto* t_data = T.data<float>();
          auto* gi_data = gi->template mutable_data<float>();
          for (int i = 0; i < N; i++) {
            auto p_mdl = x_data[i];
            auto p_emp = t_data[i];
            auto p_avg = (p_mdl + p_emp) / 2.;
            auto g_jsd = (logit(p_mdl) - logit(p_avg)) / 2.;
            gi_data[i] = go_data[i] * g_jsd;
          }
          return true;
        */
    }
}
