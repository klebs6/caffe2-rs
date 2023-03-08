crate::ix!();

pub struct MeanRangeReducerGradient<T,Context> {
    phantomA: PhantomData<T>,
    phantomB: PhantomData<Context>,
}

impl<T,Context> MeanRangeReducerGradient<T,Context> {
    
    /*
      const T* segment_grad, // GO
      T* data_grad, // GI
      const T* /*data_in*/, // I
      const T* /*data_out*/, // O
      */
          #[inline] pub fn invoke(&mut self, 
              block_size:   i64,
              blocks:       i64,
              segment_grad: *const T,
              data_grad:    *mut T,
              data_in:      *const T,
              data_out:     *const T,
              context:      *mut Context)  {
        
        todo!();
        /*
            const auto in_grad = 1.0 / blocks;
        for (int j = 0; j < block_size; ++j) {
          const T out_grad = *(segment_grad++);
          for (int i = 0; i < blocks; ++i) {
            auto idx = i * block_size + j;
            data_grad[idx] = out_grad * in_grad;
          }
        }
        */
    }
}
