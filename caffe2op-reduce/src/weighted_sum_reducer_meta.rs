crate::ix!();

pub struct WeightedSumReducerMeta<T> {
    base:       BaseReducerMeta,
    scalars:    *const T,
    first_dim:  bool,
}

impl<T> WeightedSumReducerMeta<T> {
    
    pub fn new(first: Option<bool>) -> Self {
    
        let first: bool = first.unwrap_or(true);

        todo!();
        /*
            : first_dim(first)
        */
    }
    
    #[inline] pub fn observe_input(&mut self, 
        input:     i32,
        value:     &Tensor,
        skip_dims: i32)  {

        todo!();
        /*
            if (input == 1) {
                CAFFE_ENFORCE_EQ(
                    skip_dims, value.dim(), "SCALARS mustn't have extra dimensions");
                scalars = value.data<T>();
                return;
            }
            BaseReducer::Meta::observeInput(input, value, skip_dims);
        */
    }
}
