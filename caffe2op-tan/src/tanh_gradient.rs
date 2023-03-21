crate::ix!();

pub struct TanhGradientFunctor<Context> { 
    phantom: PhantomData<Context>,
}

num_inputs!{TanhGradient, 2}

num_outputs!{TanhGradient, 1}

identical_type_and_shape_of_input!{TanhGradient, 1}

allow_inplace!{TanhGradient, vec![(1, 0)]}

impl<Context> TanhGradientFunctor<Context> {
    
    #[inline] pub fn forward<T>(&self, 
        y_dims:  &Vec<i32>,
        dy_dims: &Vec<i32>,
        y:       *const T,
        dy:      *const T,
        dx:      *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
        
        */
    }
}
