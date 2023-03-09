crate::ix!();

impl StumpFuncIndexOp<f32, i64, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& in = Input(0);
      const float* in_data = in.template data<float>();

      int lo_cnt = 0;
      for (int i = 0; i < in.numel(); i++) {
        lo_cnt += (in_data[i] <= threshold_);
      }
      auto* out_lo = Output(0, {lo_cnt}, at::dtype<int64_t>());
      auto* out_hi = Output(1, {in.numel() - lo_cnt}, at::dtype<int64_t>());
      int64_t* lo_data = out_lo->template mutable_data<int64_t>();
      int64_t* hi_data = out_hi->template mutable_data<int64_t>();
      int lidx = 0;
      int hidx = 0;
      for (int i = 0; i < in.numel(); i++) {
        if (in_data[i] <= threshold_) {
          lo_data[lidx++] = i;
        } else {
          hi_data[hidx++] = i;
        }
      }
      return true;
        */
    }
}
