crate::ix!();

pub struct SumReducerGradient<T,Context> {
    base:   BaseReducerGradient,
    s_grad: *const T,

    /*
       using FixedDispatch = FixedValues<1>;
       */
    phantom: PhantomData<Context>,
}

impl<T,Context> SumReducerGradient<T,Context> {

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
            if (FixedSize == 1) { // static if
          *data_grad = *s_grad_;
        } else if (meta.first_dim) {
          context->template CopySameDevice<T>(meta.block_size, s_grad_, data_grad);
        } else {
          math::Set<T, Context>(length, s_grad_[offset], data_grad, context);
        }
        */
    }
}
