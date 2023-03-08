crate::ix!();

pub struct SumRangeReducerGradient<T,Context> {
    phantomA: PhantomData<T>,
    phantomB: PhantomData<Context>,
}

impl<T,Context> SumRangeReducerGradient<T,Context> {

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
            // do we have some op that does it smartly with minimum number of memcpy?
        for (int64_t i = 0; i < blocks; ++i) {
          context->template CopySameDevice<T>(
              block_size, segment_grad, data_grad + block_size * i);
        }
        */
    }
}
