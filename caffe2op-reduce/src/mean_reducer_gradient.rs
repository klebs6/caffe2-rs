crate::ix!();

pub struct MeanReducerGradient<T,Context> {
    base: BaseReducerGradient,

    s_grad:  *const T,

    /*
       using FixedDispatch = FixedValues<1>;
       */
    phantom: PhantomData<Context>,
}

impl<T,Context> MeanReducerGradient<T,Context> {
    
    #[inline] pub fn compute_length() -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    pub fn new(
        meta:    &BaseReducerGradientMeta,
        s_grad:  *const T,
        context: *mut CPUContext) -> Self {

        todo!();
        /*
            : s_grad_(s_grad)
        */
    }
    
    #[inline] pub fn fill_grad<const FixedSize: i32>(&mut self, 
        meta:      &BaseReducerGradientMeta,
        data_grad: *mut T,
        offset:    i64,
        context:   *mut Context,
        length:    i32)  {
    
        todo!();
        /*
            CAFFE_ENFORCE_GT(length, 0, "Segment length must be > 0");
        if (meta.first_dim) {
          math::ScaleFixedSize<T, CPUContext, FixedSize>(
              meta.block_size, 1.0 / length, s_grad_, data_grad, context);
        } else {
          math::Set<T, CPUContext>(
              length, s_grad_[offset] * 1.0f / length, data_grad, context);
        }
        */
    }
}
