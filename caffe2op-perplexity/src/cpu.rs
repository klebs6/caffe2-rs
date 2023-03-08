crate::ix!();

impl<T,Context> PerplexityOp<T, Context> {
    
    #[inline] pub fn run_f32_on_cpu_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);

      DCHECK_EQ(X.dim(), 1);
      int N = X.dim32(0);

      auto* Y = Output(0, vector<int64_t>(), at::dtype<float>());
      const auto* Xdata = X.data<float>();

      float perplexity = 1.0;
      for (int i = 0; i < N; ++i) {
        perplexity *= pow(Xdata[i], -1.0/N);
      }
      *(Y->template mutable_data<float>()) = perplexity;
      return true;
        */
    }
}
