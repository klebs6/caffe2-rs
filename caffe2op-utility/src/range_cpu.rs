crate::ix!();

impl RangeOp<CPUContext> {
    
    #[inline] pub fn do_run_on_device<T>(
        &mut self, 
        start:  &T,
        step:   &T,
        output: *mut Tensor) -> bool 
    {
        todo!();
        /*
            auto* output_data = output->template mutable_data<T>();
      for (int i = 0; i < output->numel(); ++i) {
        output_data[i] = i * step + start;
      }
      return true;
        */
    }
}
