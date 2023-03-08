crate::ix!();

/**
  | Put forward and backward in the same
  | template?
  |
  */
pub struct SumReducer<T> {

    base:    BaseReducer,
    context: CPUContext,

    current_size:  i32,
    out:           *mut T,
    /*
       using FixedDispatch = FixedValues<1>;
       */
}

impl<T> SumReducer<T> {
    
    pub fn new(
        meta:    &BaseReducerMeta,
        out:     *mut T,
        context: *mut CPUContext) -> Self {

        todo!();
        /*
            : current_size_(0), out_(out) 

        // add a wrapper in Context for it
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
              meta.block_size, in, out_ + current_size_++, context);
        }
        */
    }
}
