crate::ix!();

pub struct MeanReducer<T, CPUContext> {
    base:              BaseReducer,
    out:               *mut T,
    current_size:      i32,

    /**
      | using FixedDispatch = FixedValues<1>;
      |
      */
    phantomCPUContext: PhantomData<CPUContext>,
}

impl<T, CPUContext> MeanReducer<T, CPUContext> {
    
    pub fn new(
        meta:    &BaseReducerMeta,
        out:     *mut T,
        context: *mut CPUContext) -> Self {

        todo!();
        /*
            : out_(out), current_size_(0) 

        if (meta.first_dim) {
          memset(out, 0, sizeof(T) * meta.block_size);
        }
        */
    }
    
    #[inline] pub fn process<const FixedSize: i32>(&mut self, 
        meta:    &BaseReducerMeta,
        input:   *const T,
        offset:  i64,
        context: *mut CPUContext)  {

        todo!();
        /*
            if (meta.first_dim) {
          math::AxpyFixedSize<T, CPUContext, FixedSize>(
              meta.block_size, 1, in, out_, context);
        } else {
          math::Sum<T, CPUContext>(
              meta.block_size, in, out_ + current_size_, context);
        }
        current_size_++;
        */
    }
    
    #[inline] pub fn finish<const FixedSize: i32>(
        &mut self, 
        meta: &BaseReducerMeta, 
        context: *mut CPUContext)  {
    
        todo!();
        /*
            if (meta.first_dim) {
          if (current_size_ > 0) {
            math::ScaleFixedSize<T, CPUContext, FixedSize>(
                meta.block_size, 1.0 / current_size_, out_, out_, context);
          }
        } else {
          math::ScaleFixedSize<T, CPUContext, FixedSize>(
              current_size_, 1.0 / meta.block_size, out_, out_, context);
        }
        */
    }
}
