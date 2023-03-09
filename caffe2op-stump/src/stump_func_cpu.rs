crate::ix!();

impl StumpFuncOp<f32, f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& in = Input(0);
      const float* in_data = in.template data<float>();

      auto* out = Output(0, in.sizes(), at::dtype<float>());
      float* out_data = out->template mutable_data<float>();
      for (int i = 0; i < in.numel(); i++) {
        out_data[i] = (in_data[i] <= threshold_) ? low_value_ : high_value_;
      }
      return true;
        */
    }
}
