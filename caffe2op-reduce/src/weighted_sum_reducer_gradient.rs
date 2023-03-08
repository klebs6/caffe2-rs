crate::ix!();

pub struct WeightedSumReducerGradient<T,Context> {
    base: BaseReducerGradient,

    s_grad:  *const T,

    /**
      | using FixedDispatch = FixedValues<1>;
      |
      */
    phantom: PhantomData<Context>,
}

impl<T,Context> WeightedSumReducerGradient<T,Context> {

    /**
      | which of the original inputs are required
      | for gradient computation
      |
      */
    #[inline] pub fn original_inputs() -> [i32; 1] {
        todo!();
        /*
           return {{1}};
           */

    }
    
    #[inline] pub fn num_aux_inputs_with_grads(def: &OperatorDef) -> i32 {
        todo!();
        /*
            return GetFlagArgument(def, "grad_on_weights");
        */
    }
    
    #[inline] pub fn requires_data_input(def: &OperatorDef) -> bool {
        
        todo!();
        /*
            return numAuxInputsWithGrads(def) > 0;
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
            math::ScaleFixedSize<T, CPUContext, FixedSize>(
            meta.block_size, meta.scalars[offset], s_grad_, data_grad, context);
        */
    }

    /**
      | Special version which is called with
      | the main input too, used only if additional
      | input grad is requested
      |
      */
    #[inline] pub fn fill_grad_with_main_input<const FixedSize: i32>(&mut self, 
        meta:      &BaseReducerGradientMeta,
        data:      *const T,
        data_grad: *mut T,
        offset:    i64,
        context:   *mut Context,
        length:    i32)  {

        todo!();
        /*
            math::ScaleFixedSize<T, CPUContext, FixedSize>(
            meta.block_size, meta.scalars[offset], s_grad_, data_grad, context);
        math::Dot(
            meta.block_size, s_grad_, data, meta.scalars_grad + offset, context);
        */
    }
}
