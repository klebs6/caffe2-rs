crate::ix!();

pub struct MeanRangeReducer<T> {
    context: CPUContext,
    phantom: PhantomData<T>,
}

impl<T> MeanRangeReducer<T> {
    
    #[inline] pub fn invoke(&mut self, 
        block_size: i64,
        blocks:     i64,
        input:      *const T,
        out:        *mut T,
        context:    *mut CPUContext)  {

        todo!();
        /*
            for (int j = 0; j < block_size; ++j) {
          T avg_value = 0;
          for (int i = 0; i < blocks; ++i) {
            avg_value += in[i * block_size + j] / blocks;
          }
          *(out++) = avg_value;
        }
        */
    }
}
