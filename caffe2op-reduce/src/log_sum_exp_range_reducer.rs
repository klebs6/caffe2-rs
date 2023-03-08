crate::ix!();

/**
  | Put forward and backward in the same
  | template?
  |
  */
pub struct LogSumExpRangeReducer<T, CPUContext> {

    r:  T, // default = 1
    phantomCPUContext: PhantomData<CPUContext>,
}

impl<T, CPUContext> LogSumExpRangeReducer<T, CPUContext> {

    #[inline] pub fn invoke(&mut self, 
        block_size: i64,
        blocks:     i64,
        input:      *const T,
        out:        *mut T,
        context:    *mut CPUContext)  {

        todo!();
        /*
            for (int j = 0; j < block_size; ++j) {
          T max_value = std::numeric_limits<T>::lowest();
          for (int i = 0; i < blocks; ++i) {
            max_value = std::max(max_value, in[i * block_size + j]);
          }
          T scaled_exp_sum = 0;
          for (int i = 0; i < blocks; ++i) {
            scaled_exp_sum += std::exp(in[i * block_size + j] - max_value);
          }
          *(out++) = std::log(scaled_exp_sum) + max_value;
        }
        */
    }
}
