crate::ix!();

pub struct MaxReducerGradient<T,Context> {
    base: BaseReducerGradient,

    s_grad:  *const T,

    /*
       using FixedDispatch = FixedValues<1>;
       */
    phantom: PhantomData<Context>,
}

impl<T,Context> MaxReducerGradient<T,Context> {

    #[inline] pub fn requires_data_input(def: &OperatorDef) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    #[inline] pub fn requires_forward_output() -> bool {
        
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
    
    #[inline] pub fn fill_grad_with_main_input_and_forward_output<const FixedSize: i32>(&mut self, 
        meta:           &BaseReducerGradientMeta,
        data:           *const T,
        data_grad:      *mut T,
        forward_output: *const T,
        offset:         i64,
        context:        *mut Context,
        length:         i32)  {

        todo!();
        /*
            for (int64_t i = 0; i < meta.block_size; ++i) {
          data_grad[i] = data[i] == forward_output[i] ? s_grad_[i] : 0;
        }
        */
    }
}
