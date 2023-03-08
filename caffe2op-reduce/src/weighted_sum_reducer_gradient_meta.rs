crate::ix!();

pub struct WeightedSumReducerGradientMeta<T> {
    base:          BaseReducerGradientMeta,
    scalars:       *const T,
    scalars_grad:  *mut T,
}

impl<T> WeightedSumReducerGradientMeta<T> {
    
    /**
      | Tensor* input_grad, // optional grad
      | to populate
      |
      */
    #[inline] pub fn observe_original_input(&mut self, 
        original_input: i32,
        value:          &Tensor,
        input_grad:     *mut Tensor,
        skip_dims:      i32)  {

        todo!();
        /*
            CAFFE_ENFORCE_EQ(1, original_input);
            scalars = value.data<T>();
            if (input_grad) {
                input_grad->ResizeLike(value);
                scalars_grad = input_grad->template mutable_data<T>();
            }
        */
    }
}
