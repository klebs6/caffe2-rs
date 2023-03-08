crate::ix!();

/**
  | Put forward and backward in the same
  | template?
  |
  */
pub struct WeightedSumReducer<T> {
    base:    BaseReducer,
    context: CPUContext,
    out:     *mut T,

    /*
       using FixedDispatch = FixedValues<1>;
       */
}

impl<T> Reducer for WeightedSumReducer<T> {

    const InputCount: isize = 2;
}

impl<T> WeightedSumReducer<T> {


    pub fn new(
        meta:    &BaseReducerGradientMeta,
        out:     *mut T,
        context: *mut CPUContext) -> Self {

        todo!();
        /*
            : out_(out) 

        // do we have a wrapper for it?
        memset(out, 0, sizeof(T) * meta.block_size);
        */
    }
    
    #[inline] pub fn process<const FixedSize: i32>(&mut self, 
        meta:    &BaseReducerGradientMeta,
        input:   *const T,
        offset:  i64,
        context: *mut CPUContext)  {
    
        todo!();
        /*
            CAFFE_ENFORCE(
            meta.first_dim,
            "WeightedSumReducer implemented only for "
            "front dimensions reduction");
        math::AxpyFixedSize<T, CPUContext, FixedSize>(
            meta.block_size, meta.scalars[offset], in, out_, context);
        */
    }
}
