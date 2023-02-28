crate::ix!();

/**
  | Computes the Jensen-Shannon divergence
  | (JSD) between two Bernoulli distributions
  | where each is parametrized by a single
  | probability.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct BernoulliJSDOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

num_inputs!{BernoulliJSD, 2}

num_outputs!{BernoulliJSD, 1}

inputs!{BernoulliJSD, 
    0 => ("X", "array of probabilities for prediction"),
    1 => ("T", "array of probabilities for target")
}

outputs!{BernoulliJSD, 
    0 => ("L", "array of JSD losses")
}

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct BernoulliJSDGradientOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

num_inputs!{BernoulliJSDGradient, 3}

num_outputs!{BernoulliJSDGradient, 1}

#[inline] pub fn klog_threshold() -> f32 {
    1e-20
}

#[inline] pub fn logit(p: f32) -> f32 {
    
    todo!();
    /*
      // it computes log(p / (1-p))
      // to avoid numeric issue, hard code p log(p) when p approaches 0
      float x = std::min(std::max(p, kLOG_THRESHOLD()), 1 - kLOG_THRESHOLD());
      return -log(1. / x - 1.);
    */
}

#[inline] pub fn entropy(p: f32) -> f32 {
    
    todo!();
    /*
      if (p < kLOG_THRESHOLD() || 1 - p < kLOG_THRESHOLD()) {
        return 0.;
      } else {
        float q = 1 - p;
        return -p * log(p) - q * log(q);
      }
    */
}


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

register_cpu_operator!{BernoulliJSD,         BernoulliJSDOp<f32, CPUContext>}

register_cpu_operator!{BernoulliJSDGradient, BernoulliJSDGradientOp<f32, CPUContext>}

pub struct GetBernoulliJSDGradient {}

impl GetGradientDefs for GetBernoulliJSDGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "BernoulliJSDGradient",
            "",
            vector<string>{GO(0), I(0), I(1)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{BernoulliJSD, GetBernoulliJSDGradient}
