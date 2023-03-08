crate::ix!();

pub struct MaxRangeReducer<T, CPUContext> {
    phantom:           PhantomData<T>,
    phantomCPUContext: PhantomData<CPUContext>,
}

impl<T, CPUContext> MaxRangeReducer<T, CPUContext> {
    
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
          *(out++) = max_value;
        }
        */
    }
}
